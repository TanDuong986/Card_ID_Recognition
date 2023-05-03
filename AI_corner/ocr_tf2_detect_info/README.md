# Localize the needed text in ID card in fine-tuning pretrained tf2 object detection model

First of all, you need to modify all the path at

```
pretrained/pipeline.config
2 files .ipynb (jupyter files)
```

to your local machine brfore running, then navigate to ```src``` folder to run

```
chmod +x prepare_dataset.sh
./prepare_dataset.sh
```

This task will create the defined structured dataset, which use for traning and testing later

Run ```recognition.ipynb``` to traning (don't forget to correct path)
Run ```test_recognition.ipynb``` to testing (don't forget to correct path)

To monitor traning job progress using TensorBoard. Run the following command:

Navigate to root

```
tensorboard --logdir=./ckpt
```

The above command will start a new TensorBoard server, which (by default) listens to port 6006 of your machine. Assuming that everything went well, you should see a print-out similar to the one below (plus/minus some warnings):

```
TensorBoard 2.2.2 at http://localhost:6006/ (Press CTRL+C to quit)
```

Enjoy!!!
