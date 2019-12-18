# ESA SmartCrop PiSchool autumn 2019 


SmartCrop is build upon the [Hugin framework](https://github.com/sage-group/hugin/tree/0.1.x) version 0.1.x. 
Hugin is a Python framework designed to help the scientists run Machine Learning experiments on geospatial raster data.

Current extensions include:

* Z Score standardization performed over entire training set/per channel before training
* Transfer weights without including last classsification layer
* Tiling the image without the requirement of having all the input images of the same size
* Include U-Net model topology
* Include a proposed implementation for HSN model and W-Net model
* Include Hugin configuration files for both training and prediction phases for U-Net, HSN and W-Net
* Add prediction metric computation for multi-class semantic segmentation



This is a proof of concept. The above mentioned extensions are going to be included in the new Hugin release. Please visit the [Wiki page](https://github.com/PiSchool/esa-crops-github/wiki) for a detailed information about the content of this repository.

Additional documentation for Hugin is available at https://hugin-eo.readthedocs.io/



## Acknowledgments
This project was carried out under the supervision of the following stakeholders ESA, E-Geos, UrbyetOrbit, MEEO (Italy) and SISTEMA (Austria) .

Hugin project development is supported by the European Space Agency through the ML4EO project.

