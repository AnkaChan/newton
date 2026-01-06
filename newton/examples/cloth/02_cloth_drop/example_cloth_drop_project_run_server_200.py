# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Cloth Drop
#
# This simulation demonstrates dropping a cloth onto a cylinder collider
# using the Simulator base class from M01_Simulator.
#
###########################################################################

import itertools
import os

import numpy as np
import tqdm
import warp as wp

from newton import ParticleFlags
from newton.examples.cloth.M01_Simulator import Simulator, default_config, get_config_value, read_obj
from example_cloth_drop_project import *
# =============================================================================
# Configuration
# =============================================================================

example_config["cloths"]["main_cloth"]["num_layers"] = 200
example_config["self_contact_margin"] = 0.32
example_config["output_path"] = "/home/horde/Code/Output/ClothDrop"

if __name__ == "__main__":
    from datetime import datetime

    # wp.clear_kernel_cache()

    # Create output folder with date/time and layer count
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_layers = example_config["cloths"]["main_cloth"]["num_layers"]
    subfolder = f"{timestamp}"
    example_config["output_path"] = os.path.join(example_config["output_path"], f"{num_layers}_layers", subfolder)

    # Create output directory
    os.makedirs(example_config["output_path"], exist_ok=True)

    # Save the run config
    save_config(example_config, example_config["output_path"])

    # Create and run the simulation
    sim = ClothDropSimulator(example_config)
    sim.finalize()
    sim.simulate()
