#!/usr/bin/env python3
import os
import sys
import logging
from pathlib import Path
from argparse import ArgumentParser

logging.basicConfig(level=logging.DEBUG)

# --- Import Pegasus API -----------------------------------------------------------
from Pegasus.api import *

class NetpredictWorkflow():
    wf = None
    sc = None
    tc = None
    rc = None
    props = None

    dagfile = None
    wf_name = None
    wf_dir = None
    
    # --- Init ---------------------------------------------------------------------
    def __init__(self, dagfile="workflow.yml"):
        self.dagfile = dagfile
        self.wf_name = "netpredict-wf"
        self.wf_dir = Path(__file__).parent.resolve()
        return
    
    # --- Write files in directory -------------------------------------------------
    def write(self):
        self.sc.write()
        self.props.write()
        self.rc.write()
        self.tc.write()
        self.wf.write()
        return


    # --- Configuration (Pegasus Properties) ---------------------------------------
    def create_pegasus_properties(self):
        self.props = Properties()
        #self.props["pegasus.monitord.encoding"] = "json"
        #self.props["pegasus.integrity.checking"] = "none" 
        return


    # --- Site Catalog -------------------------------------------------------------
    def create_sites_catalog(self, exec_site_name="condorpool"):
        self.sc = SiteCatalog()

        shared_scratch_dir = os.path.join(self.wf_dir, "scratch")
        local_storage_dir = os.path.join(self.wf_dir, "output")

        local = (Site("local")
            .add_directories(
                Directory(Directory.SHARED_SCRATCH, shared_scratch_dir)
                    .add_file_servers(FileServer("file://" + shared_scratch_dir, Operation.ALL)),
                Directory(Directory.LOCAL_STORAGE, local_storage_dir)
                    .add_file_servers(FileServer("file://" + local_storage_dir, Operation.ALL))
            )
        )

        exec_site = (Site("condorpool")
            .add_condor_profile(universe="vanilla")
            .add_pegasus_profile(
                style="condor",
                data_configuration="condorio"
            )
        )

        self.sc.add_sites(local, exec_site)
        return


    # --- Transformation Catalog (Executables and Containers) ----------------------
    def create_transformation_catalog(self):
        self.tc = TransformationCatalog()

        # Add the csv to hdf5 conversion
        csv_to_hdf5 = Transformation("csv_to_hdf5", site="local", pfn=os.path.join(self.wf_dir, "bin/csv_to_hdf5.py"), is_stageable=True)
        # Add the DDCRNN train executable
        ddcrnn_train = Transformation("ddcrnn_train", site="local", pfn=os.path.join(self.wf_dir, "bin/ddcrnn_train.sh"), is_stageable=True)
        
        self.tc.add_transformations(csv_to_hdf5, ddcrnn_train)
        return


    # --- Replica Catalog ----------------------------------------------------------
    def create_replica_catalog(self, input_data):
        self.rc = ReplicaCatalog()

        self.rc.add_replica("local", "ddcrnn_package.tar.gz", os.path.join(self.wf_dir, "bin", "ddcrnn_package.tar.gz"))
        self.rc.add_replica("local", "ddcrnn_config.yaml", os.path.join(self.wf_dir, "config", "ddcrnn_config.yaml"))
        self.rc.add_replica("local", "adj_mat.pkl", os.path.join(self.wf_dir, "config", "adj_mat.pkl"))
        self.rc.add_replica("local", input_data.rsplit("/")[-1], os.path.join(self.wf_dir, input_data))
        return

    
    # --- Create Workflow ----------------------------------------------------------
    def create_workflow(self, input_data):
        self.wf = Workflow(self.wf_name, infer_dependencies=True)
        
        # Add a CSV to HDF5 conversion job
        input_data_file = File(input_data.rsplit("/")[-1])
        speed_data_file = File("speed_data.h5")
        convert_csv_to_hdf5 = ( Job("csv_to_hdf5")
            .add_args("-i", input_data_file, "-o", speed_data_file)
            .add_inputs(input_data_file)
            .add_outputs(speed_data_file, stage_out=True, register_replica=False)
        )
        
        # Add DDCRNN train job
        ddcrnn_config_file = File("ddcrnn_config.yaml")
        links_matrix_file = File("adj_mat.pkl")
        ddcrnn_package_file = File("ddcrnn_package.tar.gz")
        model_file = File("model.tar.gz")
        results_file = File("results.tar.gz")
        ddcrnn_train = ( Job("ddcrnn_train")
            .add_args("--config_filename", ddcrnn_config_file)
            .add_inputs(ddcrnn_package_file, ddcrnn_config_file, links_matrix_file, speed_data_file)
            .add_outputs(model_file, results_file, stage_out=True, register_replica=False)
            .add_pegasus_profile(
                cores = "8",
                gpus = "1"
            )
        )
        
        self.wf.add_jobs(convert_csv_to_hdf5, ddcrnn_train)
        return


if __name__ == '__main__':
    parser = ArgumentParser(description="Pegasus Netpredict Workflow")

    parser.add_argument("-d", "--data", metavar="STR", type=str, help="Input CSV data", required=True)
    parser.add_argument("-o", "--output", metavar="STR", type=str, default="workflow.yml", help="Output file (default: workflow.yml)")

    args = parser.parse_args()
    
    workflow = NetpredictWorkflow(args.output)

    print("Creating execution sites...")
    workflow.create_sites_catalog()

    print("Creating workflow properties...")
    workflow.create_pegasus_properties()
    
    print("Creating transformation catalog...")
    workflow.create_transformation_catalog()

    print("Creating replica catalog...")
    workflow.create_replica_catalog(args.data)

    print("Creating Netpredict workflow's dag...")
    workflow.create_workflow(args.data)

    workflow.write()
