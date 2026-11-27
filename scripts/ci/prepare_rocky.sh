#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

shca_library_path() {
    ldconfig -p | awk '$1 == "libshca.so" && !found { print $NF; found = 1 }'
}

prepare_rocky_build_environment() {
    # shellcheck source=/dev/null
    source /etc/os-release
    [[ "$ID" == rocky && "$VERSION_ID" == 8.6 ]] || {
        echo "ERROR: expected Rocky Linux 8.6, got $ID $VERSION_ID" >&2; return 1;
    }

    local mirror="${ROCSHMEM_CI_ROCKY_REPO:-}" directory repo
    local -a yum_options=(--setopt=install_weak_deps=False)
    if [[ -n "$mirror" ]]; then
        echo "Rocky Linux yum mirror: $mirror"
        for directory in BaseOS AppStream extras PowerTools; do
            repo="${directory,,}"
            yum_options+=(
                "--setopt=$repo.mirrorlist="
                "--setopt=$repo.metalink="
                "--setopt=$repo.baseurl=${mirror%/}/$directory/x86_64/os/"
            )
        done
    fi
    yum "${yum_options[@]}" install -y \
        git ca-certificates binutils numactl-devel tar gzip "$@"

    # SHCA supplies its own verbs development files.
    if ! rpm -q shca-tools; then
        : "${SHCA_TOOLS_RPM_BASE_URL:?missing repository variable SHCA_TOOLS_RPM_BASE_URL}"
        local shca_rpm=/tmp/shca-tools-0.0.1_rocky8.10-1.x86_64.rpm
        local shca_rpm_url="${SHCA_TOOLS_RPM_BASE_URL%/}/${shca_rpm##*/}"
        yum "${yum_options[@]}" install -y curl
        curl --fail --location --retry 3 --proto '=http,https' \
            --proto-redir '=http,https' --output "$shca_rpm" -- "$shca_rpm_url"
        # Only the RPM tested on Rocky 8.6 may bypass its 8.10 install scripts.
        printf '%s  %s\n' 612619ad3f57f7be31117c3c926ccecde1a447e3cf9d466d1506e31cec8241d1 "$shca_rpm" |
            sha256sum --check
        yum "${yum_options[@]}" install -y \
            libev libnl3 libnl3-cli libnl3-devel libuv libwebsockets mosquitto tcp_wrappers-libs
        yum "${yum_options[@]}" remove -y --setopt=clean_requirements_on_remove=False \
            libibverbs libibumad librdmacm infiniband-diags libibverbs-utils rdma-core-devel
        rpm -ivh --nopre --nopost "$shca_rpm"
    fi

    local shca_library shca_dependencies
    ldconfig
    shca_library="$(shca_library_path)"
    if [[ ! -s /usr/include/infiniband/shca_dv.h || ! -s "$shca_library" ]]; then
        echo 'ERROR: SHCA development header or libshca.so is missing' >&2
        return 1
    fi
    shca_dependencies="$(ldd "$shca_library")"
    if grep -F 'not found' <<< "$shca_dependencies"; then
        echo 'ERROR: SHCA shared library dependencies are missing' >&2
        return 1
    fi
}
