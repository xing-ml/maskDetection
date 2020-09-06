# import necessary packages
from keras.models import load_model
import coremltools
import argparse
import pickle
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
args = vars(ap.parse_args())

#class labels
class_labels = ["Mask", "No Mask"]

# load the trained convolutional neural network
print("[INFO] loading model...")
model = load_model(args["model"])

# convert the model to coreml format
print("[INFO] converting model")
coreml_model = coremltools.converters.keras.convert(model,
	input_names="image",
	image_input_names="image",
	image_scale=1/255.0,
	class_labels=class_labels,
	is_bgr=True)

# save the model to disk
output = args["model"].rsplit(".", 1)[0] + ".mlmodel"
print("[INFO] saving model as {}".format(output))
coreml_model.save(output)