<div align="center">
    <img src='assets\logo.jpg' style="border-radius: 15px">
</div>
<br>

<h1 align="center"> MDS7201 <br> OP Wiser - Predicción de demanda </h1>


<p style="text-align: justify; line-height: 1.6; margin: 20px 0;">
    Este repositorio tiene como objetivo <strong>guardar y mostrar</strong> el desarrollo y los principales resultados obtenidos del proyecto <em>"Predicción de demanda"</em> para la empresa <strong>OP Wiser</strong>, en el curso <em>"MDS7201 - Proyecto de Ciencia de Datos"</em> del plan del Magíster en Ciencia de Datos de la Universidad de Chile.
</p>

data/
├── base/
│   ├── __init__.py
│   ├── base_dataset_maker.py
│   └── progressbar.py
├── dataset_makers/
│   ├── __init__.py
│   ├── sales_dataset_maker.py
│   └── stock_dataset_maker.py
├── data_integrator/
│   ├── __init__.py
│   └── data_integrator.py
├── data_extractors/
│   ├── __init__.py
│   └── exogenous_data_extractor.py
├── utils/
│   ├── __init__.py
│   └── data_downloader.py     # Módulo para la descarga de datos desde APIs
└── main.py
