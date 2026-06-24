import json
from pathlib import Path

from datagenic import Configs, Generator, GraphDependencies

# The native input to Configs is the neutral IR. This sample IR is the frozen
# output the Opensee `dms_to_ir` adaptor produced for the sample DMS; feeding it
# directly keeps datagenic standalone (the adaptor path is tested in opensee-core).
SOLUTIONS_IR = "samples/ir/solutions_ir.json"


def test_data_generator_column_generator():
    ######## parameters ###########
    storage_type = "local"
    output_format = "parquet"
    output_folder_path = "samples/data_generator/solutions_dms_fake_data"
    generator_config_file_path = (
        "samples/data_generation_config/data_generator_config_columns_generator.json"
    )
    ###############################

    ir = json.loads(Path(SOLUTIONS_IR).read_text())

    # create config file from the native IR
    Configs(
        input=ir,
        storage_type=storage_type,
        output_format=output_format,
        output_folder_path=output_folder_path,
    ).dump(file_path=generator_config_file_path)

    # generate data
    Generator(data_generator_config_file=generator_config_file_path).generate()


def test_data_generator_post_processor():
    # generate data
    Generator(
        data_generator_config_file="samples/data_generation_config/data_generator_config_post_processor.json"
    ).generate()


def test_data_generator_graph_dependencies():
    GraphDependencies(
        "samples/data_generation_config/data_generator_config_post_processor.json"
    ).dump_file(mermaid_header=False)


test_data_generator_post_processor()
