from gaussian_splatting.Method.render import renderTrainGS, renderGSResult


def demo_train():
    output_folder_path = "../gaussian-splatting/output/PolyTech_fine/"
    port = 6007

    renderTrainGS(output_folder_path, port)
    return True


def demo_result():
    output_folder_path = "../gaussian-splatting/output/PolyTech_fine/"
    iteration = None

    renderGSResult(output_folder_path, iteration)
    return True
