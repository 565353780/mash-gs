from mash_gs.Method.render import renderTrainGS, renderGSResult


def demo_train():
    output_folder_path = "../mash-gs/output/PolyTech_fine/"
    port = 6007

    renderTrainGS(output_folder_path, port)
    return True


def demo_result():
    output_folder_path = "../mash-gs/output/PolyTech_fine/"
    iteration = None

    renderGSResult(output_folder_path, iteration)
    return True
