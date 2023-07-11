import anvil.server
import apply_model as am
import load_model as lm

import anvil.media

anvil.server.connect("server_BIYZVIPX37WBSZOYRUXPUJX7-KMRUXYJCMVKNL6AF")

@anvil.server.callable
def segment_image(file):
    file_name = "target"
    with open(file_name, "wb") as f:
        f.write(file.get_bytes())
    original_img = am.load_image_as_tensor(file_name)
    original_shape = original_img.shape
    img, _ = lm.normalize(original_img, original_img)
    am.make_single_prediction(img, original_shape = original_shape, original_image = original_img)

    dest_path = "/home/ec2-user/Documents/Repos/TensorFlow_Tutorials/plots/img_and_mask.png"
    return anvil.media.from_file(dest_path)

anvil.server.wait_forever()