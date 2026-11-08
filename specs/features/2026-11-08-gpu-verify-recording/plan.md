# Plan — GPU verification record (v0.7.1 gate)

No GPU is needed to write this spec; steps 5–8 need a rented device.

## 1. Prepare the artefacts (CPU only, free)

1. Export a segmentation model at 432 with `deploy/export_segmentation.py`.
2. Build a 1000-frame test video with `ffmpeg`.

## 2. Prepare the box

3. Upgrade the rented instance to driver 580 so CUDA 13.x can initialise.
4. Pull `nvcr.io/nvidia/tensorrt:${NGC_CONTAINER_TAG}-py3`.
5. Start the container with the repo, the staged DALI tree, the model and the video bind-mounted.

## 3. Run the gate

6. Execute `scripts/gate_in_container.sh`, which installs the media/toolchain packages the image
   omits, exposes the image's TensorRT as a `TENSORRT_ROOTDIR` prefix, and launches
   `scripts/run_gate.sh`.
7. Arm a shutdown watchdog on the host, since a container cannot stop the instance.

## 4. Record and stop

8. Copy `~/gate-results/` off the instance, then delete it — hyperstack instances do not support
   `brev stop`, only delete.
9. Write the measured numbers into `validation.md`, including anything that did not run.
