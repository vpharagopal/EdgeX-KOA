**EdgeX-KOA: Experimental Code and Reproducibility**
**Manuscript title: EdgeX-KOA: An Empirical Evaluation of Transfer Learning and Edge Inference for Knee Osteoarthritis Grading**  
**Authors: V.P.Hara Gopal and Dr.S.Nagaraju, Department of Computer Science, Pondicherry University, Puducherry, India.**

**Overview:**
This repository contains Python scripts associated with the EdgeX-KOA study of five-class Kellgren–Lawrence (KL-0 to KL-4) knee osteoarthritis grading. The scripts cover ImageNet- and RadImageNet-initialized CNN experiments, a cross-entropy comparison, lightweight-model evaluation, Raspberry Pi profiling, and external-dataset evaluation after classification-head adaptation.
The repository provides experimental scripts; it does not imply that all experiments can be run without configuring dataset locations, obtaining the required pretrained checkpoints, or installing compatible dependencies.

**Repository contents:**
**Script	                Purpose**
`Img-DenseNet121.py`	DenseNet-121 with ImageNet initialization
`Img-ResNet50.py`	ResNet-50 with ImageNet initialization
`Img-InceptionV3.py`	InceptionV3 with ImageNet initialization
`Rad-DenseNet121.py`	DenseNet-121 with RadImageNet initialization
`Rad-ResNet50.py`	ResNet-50 with RadImageNet initialization
`Rad-InceptionV3.py`	InceptionV3 with RadImageNet initialization
`InceptionV3-CE.py`	InceptionV3 cross-entropy comparison experiment
`Edge-MobileNetV2.py`	MobileNetV2 lightweight-model experiment
`Edge-MobileNetV3.py`	MobileNetV3 lightweight-model experiment
`Edge-EfficientB0.py`	EfficientNet-B0 lightweight-model experiment
`Edge-Profiling.py`	Model complexity and Raspberry Pi CPU profiling
`External-Dataset-Validation.py`	External-dataset classification-head adaptation and hold-out evaluation

**Data:**
The primary experiments use the publicly distributed Chen/OAI-derived knee radiograph benchmark containing 8,260 images in predefined image-level partitions: 5,778 training, 826 validation, and 1,656 test images. Obtain the dataset from its original public source and comply with that source's access and use terms. The external dataset must likewise be obtained from its original source.
The datasets themselves are not included in this repository. Before execution, configure the dataset root directories, annotation/label paths, and checkpoint paths expected by each script. The predefined image-level split should be preserved when reproducing the reported primary results. Patient-wise independence was not enforced in the study's primary partition, and this limitation is discussed in the manuscript.
Environment and dependencies.
The training scripts require a compatible Python environment and the packages imported by the relevant script (including PyTorch and torchvision where applicable). Raspberry Pi profiling requires a compatible 64-bit CPU-only PyTorch runtime. Package and hardware compatibility may differ between the training workstation and Raspberry Pi.

Running the scripts:
Clone or download this repository.
Install the dependencies in the environment appropriate for the experiment.
Open each script and configure the dataset, checkpoint, and output paths for the local machine. Replace any machine-specific absolute paths before execution.
Run the desired script from a terminal, for example:
```bash
   python "Img-InceptionV3.py"
   python "Rad-InceptionV3.py"
   python "Edge-Profiling.py"
   ```
Check the script's output messages and configured output directory for predictions, metrics, logs, or profiling tables.
Run each experiment separately. Training scripts may take substantial time and may require a CUDA-compatible GPU; the Raspberry Pi profiling script is intended for the specified CPU-only device configuration. Exact execution time depends on hardware, software versions, and data storage.

**Raspberry Pi profiling notes:**
The recorded profiling protocol used a Raspberry Pi 4 Model B, CPU-only execution, batch size 1, 50 warm-up iterations, and 200 timed iterations. The saved four-model comparison reported the following means and P95 values:
Model	Inference mean (ms)	Inference P95 (ms)	End-to-end mean (ms)	End-to-end P95 (ms)	Throughput (FPS)	Peak RSS (MB)
InceptionV3	909.85	940.12	928.51	958.42	1.08	468.04
MobileNetV2	262.53	278.94	277.70	293.69	3.60	387.57
MobileNetV3	202.14	211.00	217.06	226.18	4.61	394.47
EfficientNet-B0	302.52	318.53	317.58	333.58	3.15	397.48
The recorded end-to-end profiling path includes histogram matching and CLAHE, which were not used in the primary Chen/OAI test-set preprocessing. Therefore, end-to-end times describe the extended profiling pipeline and should not be treated as an exactly matched image-to-prediction benchmark against the primary classification metrics. The saved script also reports a separate stage-wise InceptionV3 benchmark; verify and document its distinct timing procedure before interpreting it alongside the four-model comparison above.
The supplied profiling log reports that GMAC/GFLOP calculations were skipped when `thop` was unavailable. Any complexity values reported in the manuscript must therefore be supported by a separate, verified calculation or by a rerun of complexity profiling.

**Reproducibility and interpretation:**
The manuscript reports bootstrap confidence intervals calculated from saved single-run predictions. These intervals do not represent variability across independent training seeds.
Model selection was informed by comparative test-set results; analyses involving the selected reference model are described as exploratory post-hoc analyses in the manuscript.
External evaluation includes classification-head adaptation on a subset before evaluation on a disjoint hold-out subset. It is not a zero-shot evaluation.
The exported external label fields matched for the 1,314 records with both fields available; the available exports do not establish independent reader annotations, so independent inter-reader agreement could not be assessed.
Grad-CAM++ visualizations are illustrative activation maps. They have not been validated against expert-drawn anatomical regions of interest.
Actual outputs can vary with software versions, checkpoint availability, preprocessing paths, and hardware. Consult the manuscript for the experimental protocol and limitations.

**Citation:**
If you use this code, please cite the associated EdgeX-KOA manuscript once its publication details are available. Until then, cite the manuscript using the bibliographic details supplied by the authors.
