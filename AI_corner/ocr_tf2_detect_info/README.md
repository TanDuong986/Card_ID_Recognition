# Localize the needed text in ID card by fine-tuning pretrained tf2 object detection model

**Create dataset**

First of all, download data to root:

```
https://drive.google.com/drive/folders/1OL6BLQRKHdEx6mCToNgU9RQjf2thH6ha?usp=share_link
```

```
root
   |____data
   |____pretrained
   |____src
```

You need to modify all the path at

*   pretrained/pipeline.config
*   2 files .ipynb (jupyter files)

to your local machine before running, then navigate to ```src``` folder to run

```
chmod +x prepare_dataset.sh
./prepare_dataset.sh
```

**Play a Training Job**

Run ```recognition.ipynb``` to traning (don't forget to correct path)

Run ```test_recognition.ipynb``` to testing (don't forget to correct path)

**To monitor traning job progress using TensorBoard.**

Navigate to root and Run the following command:

```
tensorboard --logdir=./ckpt
```

The above command will start a new TensorBoard server, which (by default) listens to port 6006 of your machine. Assuming that everything went well, you should see a print-out similar to the one below (plus/minus some warnings):

```
TensorBoard 2.2.2 at http://localhost:6006/ (Press CTRL+C to quit)
```

**Enjoy!!!**
