/* SPDX-License-Identifier: GPL-2.0 or Linux-OpenIB */
/* Copyright (c) 2024 ZTE Corporation.  All rights reserved. */
#ifndef _ZXDH_DV_H_
#define _ZXDH_DV_H_

#include <stdio.h>
#include <stdbool.h>
#include <linux/types.h> /* For the __be64 type */
#include <sys/types.h>
#include <endian.h>
#if defined(__SSE3__)
#include <limits.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#endif /* defined(__SSE3__) */

#include <infiniband/verbs.h>
#include <infiniband/tm_types.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ZXDH_HAVE_QP_EX
#define ZXDH_HAVE_QP_EX 1
#endif

enum switch_status {
	SWITCH_CLOSE = 0,
	SWITCH_OPEN = 1,
	SWITCH_ERROR,
};

enum zxdh_qp_reset_qp_code {
	ZXDH_RESET_RETRY_TX_ITEM_FLAG = 1,
};

enum zxdh_qp_modify_qpc_mask {
	ZXDH_RETRY_CQE_SQ_OPCODE = 1 << 0,
	ZXDH_ERR_FLAG_SET = 1 << 1,
	ZXDH_PACKAGE_ERR_FLAG = 1 << 2,
	ZXDH_TX_LAST_ACK_PSN = 1 << 3,
	ZXDH_TX_LAST_ACK_WQE_OFFSET_SET = 1 << 4,
	ZXDH_TX_READ_RETRY_FLAG_SET = 1 << 5,
	ZXDH_TX_RDWQE_PYLD_LENGTH = 1 << 6,
	ZXDH_TX_RECV_READ_FLAG_SET = 1 << 7,
	ZXDH_TX_RD_MSG_LOSS_ERR_FLAG_SET = 1 << 8,
};

struct zxdh_rdma_qpc {
	uint8_t retry_flag;
	uint8_t rnr_retry_flag;
	uint8_t read_retry_flag;
	uint8_t cur_retry_count;
	uint8_t retry_cqe_sq_opcode;
	uint8_t err_flag;
	uint8_t ack_err_flag;
	uint8_t package_err_flag;
	uint8_t recv_err_flag;
	uint32_t tx_last_ack_psn;
	uint8_t retry_count;
};

struct zxdh_rdma_cap_pa {
	__u64 cap_pa_node0;
	__u64 cap_pa_node1;
};

#define CAP_NODE_NUM 2
#define NODE1 1
#define NODE0 0
#define EN_32bit_GROUP_NUM 16
#define BIT_O_31 0
#define BIT_32_63 1
#define BIT_64_95 2
#define BIT_96_127 3
#define BIT_128_159 4
#define BIT_160_191 5
#define BIT_192_223 6
#define BIT_224_255 7
#define BIT_256_287 8
#define BIT_288_319 9
#define BIT_320_351 10
#define BIT_352_383 11
#define BIT_384_415 12
#define BIT_416_447 13
#define BIT_448_479 14
#define BIT_480_511 15
#define CAP_TX 1
#define CAP_RX 2
#define FREE_TYPE_NONE 0
#define FREE_TYPE_MP 1
#define FREE_TYPE_TX 2
#define FREE_TYPE_RX 3
#define FREE_TYPE_IOVA 4
#define FREE_TYPE_HW_OBJ_DATA 5

struct zxdh_cap_cfg {
	uint8_t cap_position;
	uint64_t size;
	uint32_t channel_select[CAP_NODE_NUM];
	uint32_t channel_open[CAP_NODE_NUM];
	uint32_t node_choose[CAP_NODE_NUM];
	uint32_t node_select[CAP_NODE_NUM];
	uint32_t compare_bit_en[EN_32bit_GROUP_NUM][CAP_NODE_NUM];
	uint32_t compare_data[EN_32bit_GROUP_NUM][CAP_NODE_NUM];
	uint32_t rdma_time_wrl2d[CAP_NODE_NUM];
	uint32_t extra[CAP_NODE_NUM][EN_32bit_GROUP_NUM];
	uint32_t cap_data_start_cap;
};

#define MAX_CAP_QPS 4

struct zxdh_mp_cap_cfg {
	bool cap_use_l2d;
	uint32_t qpn[MAX_CAP_QPS];
	uint8_t qpn_num;
};

struct zxdh_mp_cap_gqp_cfg {
	bool cap_use_l2d;
	uint16_t gqpid[MAX_CAP_QPS];
	uint8_t gqp_num;
	bool is_vm_capture;
};

enum {
	MCODE_TYPE_DCQCN = 1,
	MCODE_TYPE_RTT,
	MCODE_TYPE_BASERTT,
	MCODE_TYPE_PID,
	MCODE_TYPE_WUMENG = 6,
};

struct zxdh_mp_cap_gqp {
	uint8_t mcode_type;
	uint8_t gqp_num;
	uint16_t gqpid[MAX_CAP_QPS];
	uint64_t cap_pa;
};

struct zxdh_cap_gqp {
	uint16_t gqpid[MAX_CAP_QPS];
	uint8_t gqp_num;
};

#define MAX_ACTIVE_GQP_NUM 16
struct zxdh_active_vhca_gqps {
	uint16_t vhca_id;
	uint16_t gqp_id[MAX_ACTIVE_GQP_NUM];
	uint8_t gqp_num;
};

enum zxdh_context_type {
	ZXDH_RX_READ_QPC = 1,
	ZXDH_TX_READ_QPC,
	ZXDH_READ_CQC,
	ZXDH_READ_CEQC,
	ZXDH_READ_AEQC,
	ZXDH_RX_READ_SRQC,
	ZXDH_READ_MRTE,
};

struct zxdh_context_req {
	enum zxdh_context_type type;
	__u32 resource_id;
};

#define MAX_CONTEXT_SIZE 22
struct zxdh_context_resp {
	__u64 context_info[MAX_CONTEXT_SIZE];
	__u8 context_size;
};

enum zxdh_data_type {
	ZXDH_PATH_SELECT_TYPE_CACHE = 1,
	ZXDH_PATH_SELECT_TYPE_INDICATE,
};

enum zxdh_context_type_ex {
	DATA_TYPE_PBLE_MR = 0,
	DATA_TYPE_PBLE_SQ_RQ_SRQP_SRQ_CQ_CEQ_AEQ = 1,
	DATA_TYPE_AH = 3,
	DATA_TYPE_IRD = 4,
	DATA_TYPE_TX_WINDOW = 5,
	DATA_TYPE_SQ = 11,
	DATA_TYPE_RQ = 13,
	DATA_TYPE_RQ_DOORBELL_SHADOW = 14,
	DATA_TYPE_SRQP = 15,
	DATA_TYPE_SRQ = 16,
	DATA_TYPE_SRQ_DOORBELL_SHADOW = 17,
	DATA_TYPE_CQ = 18,
	DATA_TYPE_CQ_DOORBELL_SHADOW = 19,
	DATA_TYPE_CEQ = 20,
	DATA_TYPE_AEQ = 21,
	//USER DEFINED
	DATA_TYPE_SQC = 200,
};

enum zxdh_error_code_const {
	ZXDH_NOT_SUPPORT_OBJECT_ID = 100,
	ZXDH_DMA_MEMORY_OVER_2M = 101,
	ZXDH_DMA_READ_NOT_32_ALIGN = 102,
	ZXDH_CACHE_ID_CHECK_ERROR = 103,
	ZXDH_ENTRY_IDX_ERROR = 104,
	ZXDH_PBLE_ADDRESSING_ONLY_SUPPORTS_OBJECT_NUMBER_1 = 105,
	ZXDH_NOT_SUPPORT_TWO_LEVEL_PBLE_CODE = 106,
	ZXDH_NOT_SUPPORT_VIRTUAL_ADDRESS = 107,
	ZXDH_DATA_ENTRY_IDX_OVER_LIMIT = 108,
	ZXDH_QUEUE_ID_ERROR = 109,
	/* Must be last entry*/
	ZXDH_CUSTOM_ERROR_CODE,
};

struct zxdh_get_object_data_req {
	__u32 queue_id;
	__u8 object_id;
	__u32 entry_idx;
	__u8 object_num;
};

struct zxdh_get_object_data_resp {
	__u64 object_mmap_offset;
	__u32 length;
	__u32 object_size;
	__u64 srqp_aligned_offset;
	__u16 vhca_id;
	__u8 route_id;
	__u8 reserved;
	__u32 queue_id;
};

enum zxdh_object_qp_type_const {
	ZXDH_QP_TYPE_QP = 1,
	ZXDH_QP_TYPE_SRQ,
	ZXDH_QP_TYPE_CQ,
	ZXDH_QP_TYPE_CEQ,
	ZXDH_QP_TYPE_AEQ,
};

struct zxdh_object_data_print {
	int object_id;
	const char *object_type;
	int src_path_select_type;
	__u32 length;
	__u32 queue_id;
	int qp_type;
};

#define ZXDH_L2D_MPCAP_BUFF_SIZE 0x14000u
#define ZXDH_HOST_DATA_CAP_MEM_SIZE (1024 * 1024 * 1024)

struct zxdh_cc_basic_info {
	__u32 active_gqp_cnt;
	__u16 active_vhca_sq_cnt;
	__u16 active_vhca_read_cnt;
	__u16 active_vhca_ack_cnt;
	__u16 active_qp_sq_cur_cnt;
	__u16 active_qp_rq_cur_cnt;
	__u16 task_prefetch_recv_com_cnt;
	__u64 tx_pkt_cnt;
	__u64 rx_pkt_cnt;
	__u16 flight_pkt_cnt;
	__u16 retry_timeout_cnt;
	__u16 retry_read_cnt;
	__u16 retry_rnr_cnt;
	__u16 retry_nak_cnt;
	__u16 drop_read_msg_cnt;
	__u32 tx_pkt_cnp_cnt;
	__u32 rx_pkt_cnp_cnt;
	__u32 tx_pkt_rtt_t1_cnt;
	__u32 rx_pkt_rtt_t2_cnt;
	__u32 tx_pkt_rtt_t4_cnt;
	__u32 rx_pkt_rtt_t5_cnt;
	__u16 limit_tx_sq_cnt;
	__u16 limit_tx_read_ack_cnt;
	__u32 backpres_tx_pfc_flg_pyh0_3;
	__u32 backpres_tx_pfc_flg_pyh4_7;
	__u16 backpres_tx_pfc_cnt;
	__u16 rx_pkt_ecn_cnt;
	__u8 backpres_rx_pfc_cnt;
	__u8 backpres_rx;
};

/* 端口带宽统计结构 */
struct zxdh_get_tx_rx_bw_pps {
	__u32 port_id;
	__u64 tx_bytes;
	__u64 rx_bytes;
	__u64 tx_packets;
	__u64 rx_packets;
	__u64 tx_ack_cnt;
	__u64 rx_ack_cnt;
	__u64 tx_cnp_cnt;
	__u64 rx_cnp_cnt;
	__u64 tx_rtt_t1_cnt;  /* 发送RTT请求报文计数 0x62065f8494[31:0] */
	__u64 rx_rtt_t2_cnt;  /* 接收RTT请求报文计数 0x62054004f0[31:0] */
	__u64 tx_rtt_t4_cnt;  /* 发送RTT响应报文计数 0x6205478800[25:10] */
	__u64 rx_rtt_t5_cnt;  /* 接收RTT响应报文计数 0x62054004f4[31:0] */
	__u64 timestamp;
};

enum zxdh_health_check_reg_type {
	ZXDH_NORMAL_REG,
	ZXDH_WRITE_FIRST_REG,
	ZXDH_SMMU_REG,
};

struct zxdh_health_check_req {
	__u64 reg_va;
	__u64 value_va;
	__u64 reg_value_va_ex;
	__u16 count;
	__u8 reg_type : 2;
};

struct zxdh_health_check_resp {
	__u16 count;
	__u16 count_ex;
};

struct zxdh_reg_value {
	__u64 reg_addr;
	__u32 value;
};

enum zxdh_cfg_dev_parameter_type {
	TX_STOP_ON_AEQ = 1,
	RX_STOP_ON_AEQ,
	TXRX_STOP_IOVA_CAP,
	CLEAR_ALL_CC_BASIC_CNT,
	CLEAR_ALL_GQPS_MP_CAP,
};

struct zxdh_cfg_dev_parameter_req {
	__u8 type;
	__u8 reserved1;
	__u16 reserved2;
};

struct zxdh_db_show_res_map_req {
	__u8 type;
	__u32 qp_id;
	__u64 reg_va;
	__u64 value_va;
	__u64 idx_va;
	__u32 count;
};

struct zxdh_db_show_res_map_resp {
	__u32 count;
	__u64 qp_8k_index;
};

enum zxdh_show_res_map_type {
	ZXDH_SHOW_RES_MAP_PF_TO_QPN,
	ZXDH_SHOW_RES_MAP_PF_TO_VHCA,
	ZXDH_SHOW_RES_MAP_VHCA_TO_PF,
	ZXDH_SHOW_RES_MAP_8K_TO_GQP,
	ZXDH_SHOW_RES_MAP_GQP_TO_VHCA_CREATED,
	ZXDH_SHOW_RES_MAP_GQP_TO_VHCA_ACTIVE,
	ZXDH_SHOW_RES_MAP_QP_TO_8K,
	ZXDH_SHOW_RES_MAP_UNKNOWN,
};

enum hw_module {
	HW_MODULE_TX,
	HW_MODULE_RX,
	HW_MODULE_CQP,
};

enum zxdh_ram_read_error_code_const {
	ZXDH_CUSTOM_READ_RAM_ERROR_BASE = 100,
	ZXDH_COPY_USER_PARAM_ERROR,
	ZXDH_READ_RAM_ERROR,
	ZXDH_COPY_DATA_TO_USER_ERROR,
	/* Must be last entry*/
	ZXDH_CUSTOM_ERROR_UNKOWN,
};

struct zxdh_read_ram_req {
	__u32 ram_id;
	__u32 ram_addr;
	__u32 ram_width;
	__u32 read_count;
	__u64 value_va;
	__u16 hw_module;
	__u16 reserved1;
	__u32 reserved2;
};

struct zxdh_devx_qp_init_attr {
	enum ibv_qp_type qp_type;
	struct ibv_qp_cap cap;
	__u32 flags;
	__u32 reservd[2];
};

struct zxdh_devx_init_qp_resp {
	__u32 total_size;
	__u32 rq_offset;
	__u32 shadow_offset;
};

struct zxdh_devx_qp_buffers {
	void *qp_base;
	__u32 total_size;
};

struct zxdh_verbs_qp {
#if ZXDH_HAVE_QP_EX
	union {
		struct ibv_qp qp;
		struct ibv_qp_ex qp_ex;
	};
#else
	struct ibv_qp qp;
#endif
	uint32_t comp_mask;
};

struct zxdh_devx_qp {
	struct zxdh_verbs_qp verbs_qp;
	__u64 shadow_init_value;
	__u32 qpn;
	__u32 *wqe_alloc_db;
	__u32 sqdepth;
	__u32 rqdepth;
	__u32 sqshift;
	__u32 rqshift;
	__u32 max_sq_frag_cnt;
	__u32 max_rq_frag_cnt;
	__u32 max_inline_data;
	__u16 rq_signature;
	__u16 sq_sig_all;
	__u16 shadow_size;
	__u8 rq_wqe_size_multiplier;
	__u8 sq_polarity;
	__u8 rq_polarity;
	__u8 reserved;
};

struct zxdh_devx_cq_init_attr {
	uint32_t cqe_num;
	uint32_t flags;
};

struct zxdh_devx_init_cq_resp {
	uint32_t cqe_num;
	uint32_t total_size;
	uint32_t shadow_offset;
	uint32_t reservd[2];
};

struct zxdh_devx_cq_buffers {
	void *cq_base;
	uint32_t total_size;
	uint32_t reservd[2];
};

struct zxdh_verbs_cq {
	union {
		struct ibv_cq cq;
		struct ibv_cq_ex cq_ex;
	};
};

struct zxdh_devx_cq {
	struct zxdh_verbs_cq verbs_cq;
	uint32_t cqn;
	uint32_t cqe_num;
	uint8_t polarity;
	uint32_t shadow_offset;
	uint32_t shadow_size;
};

enum switch_name_e {
	WQE_RATE_LIMIT,
	ORD_MAX_SIZE,
	SWITCH_NAME_MAX
};

enum switch_type_e {
	CONFIG_SWITCH,
	QUERY_SWITCH,
	SWITCH_TYPE_MAX
};

struct rdma_switch_param {
	char *ib_devname;
	enum switch_type_e type;
	enum switch_name_e switch_name;
	uint32_t value;
};

struct config_rdma_switch_req {
	enum switch_name_e type;
	uint32_t value;
};

struct query_rdma_switch_req {
	enum switch_name_e type;
};

struct query_rdma_switch_resp {
	uint32_t value;
	uint32_t next_reboot_value;
};

struct zxdh_ext_buf {
    void    *host_addr;
    uint64_t dev_addr;
    size_t   total_size;
};

int zxdh_get_log_trace_switch(struct ibv_context *context,
			      enum switch_status *status);
int zxdh_set_log_trace_switch(struct ibv_context *context,
			      enum switch_status status);
int zxdh_modify_qp_udp_sport(struct ibv_context *context, uint16_t udp_sport,
			     uint32_t qpn);
int zxdh_cap_start(struct ibv_context *context, struct zxdh_cap_cfg *cap_cfg,
		   struct zxdh_rdma_cap_pa *cap_pa);
int zxdh_cap_stop(struct ibv_context *context, uint8_t param);
int zxdh_cap_free(struct ibv_context *context, uint8_t param);
int zxdh_mp_capture(struct ibv_context *context,
		    struct zxdh_mp_cap_cfg *cap_mp_cfg,
		    struct zxdh_mp_cap_gqp *cap_mp_gqp);
int zxdh_mp_capture_gqp(struct ibv_context *context,
				struct zxdh_mp_cap_gqp_cfg *cfg,
				struct zxdh_mp_cap_gqp *cap_mp_gqp);
int zxdh_mp_get_data(struct ibv_context *context, uint8_t param);
int zxdh_mp_capture_clear(struct ibv_context *context,
			  struct zxdh_cap_gqp *cap_gqp);
int zxdh_get_active_vhca_gqps(struct ibv_context *context,
			      struct zxdh_active_vhca_gqps *resp);
int zxdh_get_cc_basic_info(struct ibv_context *context,
			   struct zxdh_cc_basic_info *resp);
int zxdh_query_qpc(struct ibv_qp *qp, struct zxdh_rdma_qpc *qpc);
int zxdh_modify_qpc(struct ibv_qp *qp, struct zxdh_rdma_qpc *qpc,
		    uint64_t qpc_mask);
int zxdh_reset_qp(struct ibv_qp *qp, uint64_t opcode);
int zxdh_rdma_hmc_query(struct ibv_context *context,
			struct zxdh_context_req *req,
			struct zxdh_context_resp *resp);
int zxdh_rdma_switch_query(struct ibv_context *context,
			struct query_rdma_switch_req *req,
			struct query_rdma_switch_resp *resp);
int zxdh_rdma_switch_config(struct ibv_context *context,
			struct config_rdma_switch_req *req);
int zxdh_get_object_data(struct ibv_context *context,
			 struct zxdh_get_object_data_req *req,
			 struct zxdh_get_object_data_resp *resp);
int zxdh_rdma_health_check(struct ibv_context *context,
			   struct zxdh_health_check_req *req,
			   struct zxdh_health_check_resp *resp);
int zxdh_cfg_dev_parameter(struct ibv_context *context,
			   struct zxdh_cfg_dev_parameter_req *req);
int zxdh_show_res_map(struct ibv_context *context,
		      struct zxdh_db_show_res_map_req *req,
		      struct zxdh_db_show_res_map_resp *resp);
int zxdh_read_ram(struct ibv_context *context, struct zxdh_read_ram_req *req);

int zxdh_devx_get_qp_init_size(struct ibv_context *ibv_ctx,
			       struct zxdh_devx_qp_init_attr *attr,
			       struct zxdh_devx_init_qp_resp *resp);
struct zxdh_devx_qp *zxdh_devx_create_qp(struct ibv_pd *pd,
					 struct ibv_qp_init_attr *attr,
					 struct zxdh_devx_qp_buffers *buffer);
int zxdh_devx_destroy_qp(struct zxdh_devx_qp *devx_qp);
int zxdh_devx_modify_qp(struct zxdh_devx_qp *devx_qp, struct ibv_qp_attr *attr,
			int attr_mask);
int zxdh_set_qp_credit_flag(struct ibv_context *context,
					struct ibv_qp *qp, uint64_t credit_flag);
int zxdh_devx_get_cq_init_size(struct ibv_context *context,
			       struct zxdh_devx_cq_init_attr *attr,
			       struct zxdh_devx_init_cq_resp *resp);
struct zxdh_devx_cq *zxdh_devx_create_cq(struct ibv_context *context,
					 int cqe_num,
					 struct ibv_comp_channel *channel,
					 int comp_vector,
					 struct zxdh_devx_cq_buffers *buffer);
int zxdh_devx_destroy_cq(struct zxdh_devx_cq *cq);
int zxdh_devx_modify_cq(struct ibv_cq *cq, uint32_t overflow_check_en);
struct ibv_mr *zxdh_devx_reg_mr(struct ibv_pd *pd, void *host_addr, size_t length, uint64_t dev_addr, unsigned int access);
int zxdh_devx_dereg_mr(struct ibv_mr *mr);
struct ibv_cq *zxdh_devx_create_cq_with_ext_buf(struct ibv_context *context,
	int cqe_num,
	struct ibv_comp_channel *channel,
	int comp_vector,
	struct zxdh_ext_buf *buf);
struct ibv_qp *zxdh_devx_create_qp_with_ext_buf(struct ibv_pd *pd,
	struct ibv_qp_init_attr *attr,
	struct zxdh_ext_buf *buf);
int zxdh_devx_get_peer_dev_sbdf(struct ibv_context *ibv_ctx, uint64_t *peer_pci_bdf);


#define ZXDH_MAX_RANGES_PER_GQP	16
#define ZXDH_MAX_MAPPINGS_ENTRIES	16
#define ZXDH_MAX_RANGES_PER_ENTRY	64

#define ZXDH_MAX_GQPS_PER_DEV	512

struct zxdh_qpn_range {
	uint32_t start_qpn;
	uint32_t end_qpn;
};

struct zxdh_gqp_qps_info {
	uint16_t gqp_id;
	uint16_t total_qp_num;
	uint16_t range_num;
	struct zxdh_qpn_range ranges[ZXDH_MAX_RANGES_PER_GQP];
};

struct zxdh_qp_gqp_resp {
	uint16_t gqp_id;
	uint32_t result;
	uint32_t base_qpn;  /* Valid QPN range start */
	uint32_t max_qpn;   /* Valid QPN range end (exclusive) */
};

struct zxdh_gqp_qps_resp {
	uint16_t gqp_id;
	uint16_t total_qp_num;
	uint16_t range_num;
	uint16_t result;
	struct zxdh_qpn_range ranges[ZXDH_MAX_RANGES_PER_GQP];
};

struct zxdh_all_gqp_qps_resp {
	uint16_t gqp_cnt;
	uint16_t reserved;
	struct zxdh_gqp_qps_resp gqps[ZXDH_MAX_GQPS_PER_DEV];
};

struct zxdh_get_all_gqp_qps_req {
	uint64_t buf_va;
	uint32_t buf_len;
};

int zxdh_get_qp_gqp(struct ibv_context *context,
		    uint32_t qpn,
		    struct zxdh_qp_gqp_resp *resp);

int zxdh_get_gqp_qps(struct ibv_context *context,
		     uint16_t gqp_id,
		     struct zxdh_gqp_qps_info *info);

int zxdh_get_all_gqp_qps(struct ibv_context *context,
			 struct zxdh_all_gqp_qps_resp *info);

#ifdef __cplusplus
}
#endif

#endif
