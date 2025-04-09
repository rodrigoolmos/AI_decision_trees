# Copyright (c) 2011-2024 Columbia University, System Level Design Group
# SPDX-License-Identifier: Apache-2.0

# User-defined configuration ports
# <<--directives-param-->>
set_directive_interface -mode ap_none "top" conf_info_read_inference
set_directive_interface -mode ap_none "top" conf_info_load_features
set_directive_interface -mode ap_none "top" conf_info_load_trees

# Insert here any custom directive
set_directive_dataflow "top/go"
