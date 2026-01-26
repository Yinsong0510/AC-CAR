<h1 align="center">
  <font color="purple">
    AC-CAR: Adaptive Conditional Contrast-Agnostic Deformable Image Registration with Uncertainty Estimation (IEEE Transactions on Medical Imaging)  
    <a href="https://ieeexplore.ieee.org/abstract/document/11345324" target="_blank" style="text-decoration:none; color:blue; font-size:24px;">[Paper]</a>
  </font>
</h1>

<p align="center">
  Official Implementation of the paper  
  <i>"AC-CAR: Adaptive Conditional Contrast-Agnostic Deformable Image Registration with Uncertainty Estimation", IEEE Transactions on Medical Imaging, 2026. Written by Yinsong Wang, Siyi Du, Xinzhe Luo, and Chen Qin</i>
</p>

<p align="center">
  <img src="network.png" alt="network" width="1000"/>
</p>

---

# Prerequisites
- `Python 3.9`
- `PyTorch >=1.10.1`
- `NumPy`
- `NiBabel`

# Data
Due to redistribution restrictions, we cannot share the original or processed data. The datasets are publicly available upon application at:  
- [CamCAN dataset](https://opendata.mrc-cbu.cam.ac.uk/projects/camcan/)  
- [CMRxRecon 2023 dataset](https://cmrxrecon.github.io/Home.html)
- [IXI dataset](https://brain-development.org/ixi-dataset/)

# Training
For the CamCAN dataset, run
- `python train_CamCAN.py`
  
For the CMRxRecon dataset, run
- `python train_CMR.py`

For the CMRxRecon dataset, run
- `python train_IXI.py`

> Note: You may need to customize your own dataloader. Add your customized dataloader to <code>datasets/dataloader.py</code>.

# Inference
For the CamCAN dataset, run
- `python test_CamCAN.py`
  
For the CMRxRecon dataset, run
- `python test_CMR.py`

For the CMRxRecon dataset, run
- `python test_IXI.py`

# Publication
If you make use of the code or found it useful, please cite the paper:

<p align="center">
<b>AC-CAR: Adaptive Conditional Contrast-Agnostic Deformable Image Registration with Uncertainty Estimation</b>
</p>

<p align="center">
<pre>
@article{wang2026adaptive,
  title={Adaptive Conditional Contrast-Agnostic Deformable Image Registration with Uncertainty Estimation},
  author={Wang, Yinsong and Luo, Xinzhe and Du, Siyi and Qin, Chen},
  journal={IEEE Transactions on Medical Imaging},
  year={2026},
  publisher={IEEE}
}
</pre>
</p>

---

