// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

// Demonstrate same-team ordering across streams and verify the results.
#include <rocshmem/rocshmem.hpp>
#include <hip/hip_runtime.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace rocshmem;
#define HIP_OK(x) do { auto e=(x); if(e!=hipSuccess) {fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e)); abort();} } while(0)
#define SHMEM_OK(x) do { if((x)!=0) abort(); } while(0)
__global__ void delay() { auto start=clock64(); while(clock64()-start < 20000000ULL) {} }
int main(int argc, char** argv) {
  const char* local=std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  HIP_OK(hipSetDevice(local ? std::atoi(local) : 0));
  // Exchange the bootstrap ID via MPI; use the same TCP initialization path
  // as functional tests with ROCSHMEM_TEST_UUID, without relying on PMIx.
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) abort();
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  rocshmem_uniqueid_t uid;
  if (mpi_rank == 0) SHMEM_OK(rocshmem_get_uniqueid(&uid));
  if (MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD) != MPI_SUCCESS) abort();
  rocshmem_init_attr_t attr;
  SHMEM_OK(rocshmem_set_attr_uniqueid_args(mpi_rank,mpi_size,&uid,&attr));
  SHMEM_OK(rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID,&attr));
  const int rank=rocshmem_my_pe(), pes=rocshmem_n_pes();
  constexpr int capacity=262144, rounds=12;
  int *src=static_cast<int*>(rocshmem_malloc(size_t(capacity)*pes*sizeof(int)));
  int *dst=static_cast<int*>(rocshmem_malloc(size_t(capacity)*rounds*sizeof(int)));
  std::vector<int> input(size_t(capacity)*pes, rank+1);
  HIP_OK(hipMemcpy(src,input.data(),input.size()*sizeof(int),hipMemcpyHostToDevice));
  hipStream_t streams[2];
  for(auto& s:streams) HIP_OK(hipStreamCreateWithFlags(&s,hipStreamNonBlocking));
  // Recreate the parent repeatedly to exercise event and child-team cleanup.
  for(int repeat=0;repeat<3;repeat++) {
    rocshmem_team_t team;
    SHMEM_OK(rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD,0,1,pes,nullptr,0,&team));
    HIP_OK(hipMemset(dst,0,size_t(capacity)*rounds*sizeof(int)));
    for(int i=0;i<rounds;i++) {
      auto stream=streams[std::getenv("ORDER_SINGLE_STREAM") ? 0 : i%2];
      // Alternate single-WG, multi-WG, and growth of the cached child set.
      int n=(i%3==0 ? 256 : i%3==1 ? 65536 : capacity);
      if(rank==0 && i%2==0) delay<<<1,1,0,stream>>>();
      if(i%2==0) {
        SHMEM_OK(rocshmem_reduce_scatter_on_stream(team,dst+size_t(i)*capacity,
          src,n,stream,ROCSHMEM_DATATYPE_INT,ROCSHMEM_REDUCE_SUM));
      } else {
        rocshmem_broadcastmem_on_stream(team,dst+size_t(i)*capacity,
          src,size_t(n)*sizeof(int),0,stream);
      }
      rocshmem_team_sync_on_stream(team,stream);
    }
    for(auto s:streams) HIP_OK(hipStreamSynchronize(s));
    std::vector<int> output(size_t(capacity)*rounds);
    HIP_OK(hipMemcpy(output.data(),dst,output.size()*sizeof(int),hipMemcpyDeviceToHost));
    for(int i=0;i<rounds;i++) {
      int n=(i%3==0 ? 256 : i%3==1 ? 65536 : capacity);
      int expected=i%2==0 ? pes*(pes+1)/2 : 1;
      for(int j=0;j<n;j++) if(output[size_t(i)*capacity+j]!=expected) {
        fprintf(stderr,"rank=%d repeat=%d call=%d element=%d got=%d expected=%d\n",
          rank,repeat,i,j,output[size_t(i)*capacity+j],expected); abort();
      }
    }
    // Every PE must finish inspecting dst before the next broadcast overwrites it.
    rocshmem_barrier_all();
    // A new stream can reuse a destroyed handle: the event must still order it.
    HIP_OK(hipStreamDestroy(streams[1]));
    HIP_OK(hipStreamCreateWithFlags(&streams[1],hipStreamNonBlocking));
    rocshmem_broadcastmem_on_stream(team,dst,src,256*sizeof(int),0,streams[1]);
    rocshmem_team_sync_on_stream(team,streams[1]);
    rocshmem_team_destroy(team);
    rocshmem_barrier_all();
  }
  // Cover cold capture, cached resources, and capture needing more child teams.
  const bool cold_graph=std::getenv("ORDER_COLD_GRAPH")!=nullptr;
  for(int graph_n : {256,65536,262144}) {
    rocshmem_team_t graph_team;
    SHMEM_OK(rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD,0,1,pes,nullptr,0,&graph_team));
    HIP_OK(hipMemset(dst,0,size_t(graph_n)*sizeof(int)));
    if(!cold_graph) {
      const int warm_n=graph_n==capacity ? graph_n/4 : graph_n;
      rocshmem_broadcastmem_on_stream(graph_team,dst,src,size_t(warm_n)*sizeof(int),0,streams[0]);
      HIP_OK(hipStreamSynchronize(streams[0]));
    }
    hipGraph_t graph;
    hipGraphExec_t graph_exec;
    HIP_OK(hipStreamBeginCapture(streams[0],hipStreamCaptureModeGlobal));
    // Every PE must finish clearing dst before a peer can write to it.
    rocshmem_team_sync_on_stream(graph_team,streams[0]);
    rocshmem_broadcastmem_on_stream(graph_team,dst,src,size_t(graph_n)*sizeof(int),0,streams[0]);
    SHMEM_OK(rocshmem_reduce_scatter_on_stream(graph_team,dst+capacity,src,graph_n,
      streams[0],ROCSHMEM_DATATYPE_INT,ROCSHMEM_REDUCE_SUM));
    rocshmem_team_sync_on_stream(graph_team,streams[0]);
    HIP_OK(hipStreamEndCapture(streams[0],&graph));
    HIP_OK(hipGraphInstantiate(&graph_exec,graph,nullptr,nullptr,0));
    std::vector<int> graph_output(graph_n);
    for(int replay=0;replay<2;replay++) {
      HIP_OK(hipMemsetAsync(dst,0,size_t(capacity+graph_n)*sizeof(int),streams[0]));
      HIP_OK(hipGraphLaunch(graph_exec,streams[0]));
      HIP_OK(hipStreamSynchronize(streams[0]));
      HIP_OK(hipMemcpy(graph_output.data(),dst,size_t(graph_n)*sizeof(int),hipMemcpyDeviceToHost));
      for(int j=0;j<graph_n;j++) if(graph_output[j]!=1) {
        fprintf(stderr,"graph broadcast rank=%d n=%d replay=%d element=%d got=%d expected=1\n",
          rank,graph_n,replay,j,graph_output[j]); abort();
      }
      HIP_OK(hipMemcpy(graph_output.data(),dst+capacity,size_t(graph_n)*sizeof(int),hipMemcpyDeviceToHost));
      for(int j=0;j<graph_n;j++) if(graph_output[j]!=pes*(pes+1)/2) {
        fprintf(stderr,"graph reduce-scatter rank=%d n=%d replay=%d element=%d got=%d expected=%d\n",
          rank,graph_n,replay,j,graph_output[j],pes*(pes+1)/2); abort();
      }
      rocshmem_barrier_all();
    }
    rocshmem_team_sync_on_stream(graph_team,streams[1]);
    HIP_OK(hipStreamSynchronize(streams[1]));
    HIP_OK(hipGraphExecDestroy(graph_exec));
    HIP_OK(hipGraphDestroy(graph));
    rocshmem_team_destroy(graph_team);
    if(rank==0) printf("GRAPH_CAPTURE PASS n=%d cold=%d\n",graph_n,cold_graph);
  }
  for(auto s:streams) HIP_OK(hipStreamDestroy(s));
  rocshmem_free(dst);rocshmem_free(src);
  if(rank==0) puts("SAME_TEAM_STREAM_ORDER PASS");
  rocshmem_finalize();
  MPI_Finalize();
}
