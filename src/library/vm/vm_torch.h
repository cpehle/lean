/*
 Copyright (c) 2017 Christian Pehle. All rights reserved.
 Released under Apache 2.0 license as described in the file LICENSE.
 Author: Daniel Selsam
 */
 #pragma once
 #include <torch/torch.h>
 #include <limits>
 #include "library/vm/vm.h"

namespace lean {
vm_obj to_obj(torch::Tensor const & v);
torch::Tensor const & to_torch(vm_obj const & o);

void initialize_vm_torch();
void finalize_vm_torch();
}