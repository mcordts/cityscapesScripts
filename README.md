# The Cityscapes Dataset
#O conjunto de dados de paisagens urbanas

This repository contains scripts for inspection, preparation, and evaluation of the Cityscapes dataset. This large-scale dataset contains a diverse set of stereo video sequences recorded in street scenes from 50 different cities, with high quality pixel-level annotations of 5 000 frames in addition to a larger set of 20 000 weakly annotated frames.
Este repositório contém scripts para inspeção, preparação e avaliação do conjunto de dados Cityscapes. Este conjunto de dados em grande escala contém um conjunto diversificado de sequências de vídeo estéreo gravadas em cenas de rua de 50 cidades diferentes, com anotações de nível de pixel de alta qualidade de 5.000 quadros, além de um conjunto maior de 20.000 quadros fracamente anotados.

Details and download are available at: www.cityscapes-dataset.com
Detalhes e download estão disponíveis em: www.cityscapes-dataset.com

## Dataset Structure
## Estrutura do conjunto de dados

The folder structure of the Cityscapes dataset is as follows:
A estrutura de pastas do conjunto de dados Cityscapes é a seguinte:

```
{root}/{type}{video}/{split}/{city}/{city}_{seq:0>6}_{frame:0>6}_{type}{ext}
```

The meaning of the individual elements is:
 - `root`  the root folder of the Cityscapes dataset. Many of our scripts check if an environment variable `CITYSCAPES_DATASET` pointing to this folder exists and use this as the default choice.
 - `type`  the type/modality of data, e.g. `gtFine` for fine ground truth, or `leftImg8bit` for left 8-bit images.
 - `split` the split, i.e. train/val/test/train_extra/demoVideo. Note that not all kinds of data exist for all splits. Thus, do not be surprised to occasionally find empty folders.
 - `city`  the city in which this part of the dataset was recorded.
 - `seq`   the sequence number using 6 digits.
 - `frame` the frame number using 6 digits. Note that in some cities very few, albeit very long sequences were recorded, while in some cities many short sequences were recorded, of which only the 19th frame is annotated.
 - `ext`   the extension of the file and optionally a suffix, e.g. `_polygons.json` for ground truth files

O significado dos elementos individuais é:
 - `root` a pasta raiz do conjunto de dados Cityscapes. Muitos de nossos scripts verificam se existe uma variável de ambiente `CITYSCAPES_DATASET` apontando para esta pasta e usam isso como a escolha padrão.
 - `type` o tipo/modalidade de dados, por ex. `gtFine` para dados precisos, ou `leftImg8bit` para imagens esquerdas de 8 bits.
 - `split` a divisão, ou seja, train/val/test/train_extra/demoVideo. Observe que nem todos os tipos de dados existem para todas as divisões. Portanto, não se surpreenda ao encontrar ocasionalmente pastas vazias.
 - `city` a cidade na qual esta parte do conjunto de dados foi registrada.
 - `seq` o número de sequência usando 6 dígitos.
 - `frame` o número do quadro usando 6 dígitos. Observe que em algumas cidades foram gravadas muito poucas sequências, embora muito longas, enquanto em algumas cidades foram gravadas muitas sequências curtas, das quais apenas o 19º quadro é anotado.
 - `ext` a extensão do arquivo e opcionalmente um sufixo, por exemplo `_polygons.json` para arquivos de informações básicas

Possible values of `type`
 - `gtFine`       the fine annotations, 2975 training, 500 validation, and 1525 testing. This type of annotations is used for validation, testing, and optionally for training. Annotations are encoded using `json` files containing the individual polygons. Additionally, we provide `png` images, where pixel values encode labels. Please refer to `helpers/labels.py` and the scripts in `preparation` for details.
 - `gtCoarse`     the coarse annotations, available for all training and validation images and for another set of 19998 training images (`train_extra`). These annotations can be used for training, either together with gtFine or alone in a weakly supervised setup.
 - `gtBbox3d`     3D bounding box annotations of vehicles. Please refer to [Cityscapes 3D (Gählert et al., CVPRW '20)](https://arxiv.org/abs/2006.07864) for details.
 - `gtBboxCityPersons` pedestrian bounding box annotations, available for all training and validation images. Please refer to `helpers/labels_cityPersons.py` as well as [CityPersons (Zhang et al., CVPR '17)](https://bitbucket.org/shanshanzhang/citypersons) for more details. The four values of a bounding box are (x, y, w, h), where (x, y) is its top-left corner and (w, h) its width and height.

Possíveis valores de `type`
 - `gtFine` as anotações finas, 2975 treinamento, 500 validação e 1525 teste. Esse tipo de anotação é usado para validação, teste e, opcionalmente, para treinamento. As anotações são codificadas usando arquivos `json` contendo os polígonos individuais. Além disso, fornecemos imagens `png`, onde os valores de pixel codificam rótulos. Consulte `helpers/labels.py` e os scripts em `preparation` para obter detalhes.
 - `gtCoarse` as anotações grosseiras, disponíveis para todas as imagens de treinamento e validação e para outro conjunto de 19998 imagens de treinamento (`train_extra`). Essas anotações podem ser usadas para treinamento, junto com gtFine ou sozinhas em uma configuração fracamente supervisionada.
 - `gtBbox3d` Anotações de caixa delimitadora 3D de veículos. Consulte [Cityscapes 3D (Gählert et al., CVPRW '20)](https://arxiv.org/abs/2006.07864) para obter detalhes.
 - Anotações de caixas delimitadoras de pedestres `gtBboxCityPersons`, disponíveis para todas as imagens de treinamento e validação. Consulte `helpers/labels_cityPersons.py`, bem como [CityPersons (Zhang et al., CVPR '17)](https://bitbucket.org/shanshanzhang/citypersons) para obter mais detalhes. Os quatro valores de uma caixa delimitadora são (x, y, w, h), onde (x, y) é seu canto superior esquerdo e (w, h) sua largura e altura.

 - `leftImg8bit`  the left images in 8-bit LDR format. These are the standard annotated images.
 - `leftImg8bit_blurred`  the left images in 8-bit LDR format with faces and license plates blurred. Please compute results on the original images but use the blurred ones for visualization. We thank [Mapillary](https://www.mapillary.com/) for blurring the images.
 - `leftImg16bit` the left images in 16-bit HDR format. These images offer 16 bits per pixel of color depth and contain more information, especially in very dark or bright parts of the scene. Warning: The images are stored as 16-bit pngs, which is non-standard and not supported by all libraries.
 - `rightImg8bit`  the right stereo views in 8-bit LDR format.
 - `rightImg16bit` the right stereo views in 16-bit HDR format.
 - `timestamp`     the time of recording in ns. The first frame of each sequence always has a timestamp of 0.
 - `disparity`     precomputed disparity depth maps. To obtain the disparity values, compute for each pixel p with p > 0: d = ( float(p) - 1. ) / 256., while a value p = 0 is an invalid measurement. Warning: the images are stored as 16-bit pngs, which is non-standard and not supported by all libraries.
 - `camera`        internal and external camera calibration. For details, please refer to [csCalibration.pdf](docs/csCalibration.pdf)
 - `vehicle`       vehicle odometry, GPS coordinates, and outside temperature. For details, please refer to [csCalibration.pdf](docs/csCalibration.pdf)

- `leftImg8bit`  as imagens da esquerda no formato LDR de 8 bits. Estas são as imagens anotadas padrão.
- `leftImg8bit_blurred`  as imagens da esquerda em formato LDR de 8 bits com rostos e placas desfocadas. Calcule os resultados nas imagens originais, mas use as desfocadas para visualização. Agradecemos [Mapillary](https://www.mapillary.com/) por desfocar as imagens.
 - `leftImg16bit`  as imagens da esquerda em formato HDR de 16 bits. Essas imagens oferecem 16 bits por pixel de profundidade de cor e contêm mais informações, especialmente em partes muito escuras ou claras da cena. Aviso: as imagens são armazenadas como pngs de 16 bits, o que não é padrão e não é suportado por todas as bibliotecas.
- `rightImg8bit`   as visualizações estéreo certas no formato LDR de 8 bits.
- `rightImg16bit`  as visualizações estéreo certas no formato HDR de 16 bits.
- `timestamp`      o tempo de gravação em ns. O primeiro quadro de cada sequência sempre tem um timestamp de 0.
- `disparity`      mapas de profundidade de disparidade pré-computados. Para obter os valores de disparidade, calcule para cada pixel p com p > 0: d = ( float(p) - 1. ) / 256., enquanto um valor p = 0 é uma medida inválida. Aviso: as imagens são armazenadas como pngs de 16 bits, o que não é padrão e não é suportado por todas as bibliotecas.
- `camera`         calibração de câmera interna e externa. Para obter detalhes, consulte [csCalibration.pdf](docs/csCalibration.pdf)
- `vehicle`        odometria do veículo, coordenadas de GPS e temperatura externa. Para obter detalhes, consulte [csCalibration.pdf](docs/csCalibration.pdf)


More types might be added over time and also not all types are initially available. Please let us know if you need any other meta-data to run your approach.

Mais tipos podem ser adicionados ao longo do tempo e nem todos os tipos estão disponíveis inicialmente. Informe-nos se precisar de outros metadados para executar sua abordagem.

Possible values of `split`
 - `train`       usually used for training, contains 2975 images with fine and coarse annotations
 - `val`         should be used for validation of hyper-parameters, contains 500 image with fine and coarse annotations. Can also be used for training.
 - `test`        used for testing on our evaluation server. The annotations are not public, but we include annotations of ego-vehicle and rectification border for convenience.
 - `train_extra` can be optionally used for training, contains 19998 images with coarse annotations
 - `demoVideo`   video sequences that could be used for qualitative evaluation, no annotations are available for these videos

Possíveis valores de `split`
 - `train` geralmente usado para treinamento, contém 2975 imagens com anotações finas e grosseiras
 - `val` deve ser usado para validação de hiperparâmetros, contém 500 imagens com anotações finas e grossas. Também pode ser usado para treinamento.
 - `test` usado para testes em nosso servidor de avaliação. As anotações não são públicas, mas incluímos anotações de auto-veículo e borda de retificação por conveniência.
 - `train_extra` pode ser usado opcionalmente para treinamento, contém 19998 imagens com anotações grosseiras
 - sequências de vídeo `demoVideo` que podem ser usadas para avaliação qualitativa, não há anotações disponíveis para esses vídeos

## Scripts
## Roteiros

### Installation
### Instalação
Install `cityscapesscripts` with `pip`
```
Instale `cityscapesscripts` com `pip`
```
python -m pip install cityscapesscripts
```
python -m pip instalar cityscapesscripts
```
Graphical tools (viewer and label tool) are based on Qt5 and can be installed via
```
As ferramentas gráficas (visualizador e ferramenta de etiqueta) são baseadas no Qt5 e podem ser instaladas via
```
python -m pip install cityscapesscripts[gui]
```
python -m pip install cityscapesscripts[gui]
```
### Usage
### Uso

The installation installs the cityscapes scripts as a python module named `cityscapesscripts` and exposes the following tools
- `csDownload`: Download the cityscapes packages via command line.
- `csViewer`: View the images and overlay the annotations.
- `csLabelTool`: Tool that we used for labeling.
- `csEvalPixelLevelSemanticLabeling`: Evaluate pixel-level semantic labeling results on the validation set. This tool is also used to evaluate the results on the test set.
- `csEvalInstanceLevelSemanticLabeling`: Evaluate instance-level semantic labeling results on the validation set. This tool is also used to evaluate the results on the test set.
- `csEvalPanopticSemanticLabeling`: Evaluate panoptic segmentation results on the validation set. This tool is also used to evaluate the results on the test set.
- `csEvalObjectDetection3d`: Evaluate 3D object detection on the validation set. This tool is also used to evaluate the results on the test set.
- `csCreateTrainIdLabelImgs`: Convert annotations in polygonal format to png images with label IDs, where pixels encode "train IDs" that you can define in `labels.py`.
- `csCreateTrainIdInstanceImgs`: Convert annotations in polygonal format to png images with instance IDs, where pixels encode instance IDs composed of "train IDs".
- `csCreatePanopticImgs`: Convert annotations in standard png format to [COCO panoptic segmentation format](http://cocodataset.org/#format-data).
- `csPlot3dDetectionResults`: Visualize 3D object detection evaluation results stored in .json format.

A instalação instala os scripts cityscapes como um módulo python chamado `cityscapesscripts` e expõe as seguintes ferramentas
- `csDownload`: Baixe os pacotes de paisagens urbanas via linha de comando.
- `csViewer`: Visualize as imagens e sobreponha as anotações.
- `csLabelTool`: Ferramenta que usamos para rotulagem.
- `csEvalPixelLevelSemanticLabeling`: avalia os resultados de rotulagem semântica em nível de pixel no conjunto de validação. Essa ferramenta também é usada para avaliar os resultados no conjunto de teste.
- `csEvalInstanceLevelSemanticLabeling`: avalia os resultados de rotulagem semântica em nível de instância no conjunto de validação. Essa ferramenta também é usada para avaliar os resultados no conjunto de teste.
- `csEvalPanopticSemanticLabeling`: Avalie os resultados da segmentação panóptica no conjunto de validação. Essa ferramenta também é usada para avaliar os resultados no conjunto de teste.
- `csEvalObjectDetection3d`: Avalie a detecção de objetos 3D no conjunto de validação. Essa ferramenta também é usada para avaliar os resultados no conjunto de teste.
- `csCreateTrainIdLabelImgs`: Converte anotações em formato poligonal em imagens png com IDs de rótulo, onde os pixels codificam "IDs de trem" que você pode definir em `labels.py`.
- `csCreateTrainIdInstanceImgs`: Converte anotações em formato poligonal em imagens png com IDs de instância, onde os pixels codificam IDs de instância compostos de "IDs de trem".
- `csCreatePanopticImgs`: Converte anotações em formato png padrão para [formato de segmentação COCO panóptico](http://cocodataset.org/#format-data).
- `csPlot3dDetectionResults`: Visualize os resultados da avaliação de detecção de objetos 3D armazenados no formato .json.

### Package Content
### Conteúdo do pacote

The package is structured as follows
 - `helpers`: helper files that are included by other scripts
 - `viewer`: view the images and the annotations
 - `preparation`: convert the ground truth annotations into a format suitable for your approach
 - `evaluation`: validate your approach
 - `annotation`: the annotation tool used for labeling the dataset
 - `download`: downloader for Cityscapes packages

Note that all files have a small documentation at the top. Most important files
 - `helpers/labels.py`: central file defining the IDs of all semantic classes and providing mapping between various class properties.
 - `helpers/labels_cityPersons.py`: file defining the IDs of all CityPersons pedestrian classes and providing mapping between various class properties.
 - `setup.py`: run `CYTHONIZE_EVAL= python setup.py build_ext --inplace` to enable cython plugin for faster evaluation. Only tested for Ubuntu.

O pacote está estruturado da seguinte forma
 - `helpers`: arquivos auxiliares que são incluídos por outros scripts
 - `viewer`: visualizar as imagens e as anotações
 - `preparação`: converta as anotações de verdade em um formato adequado para sua abordagem
 - `avaliação`: valide sua abordagem
 - `annotation`: a ferramenta de anotação usada para rotular o conjunto de dados
 - `download`: downloader para pacotes Cityscapes

Observe que todos os arquivos possuem uma pequena documentação na parte superior. arquivos mais importantes
 - `helpers/labels.py`: arquivo central definindo os IDs de todas as classes semânticas e fornecendo mapeamento entre várias propriedades de classe.
 - `helpers/labels_cityPersons.py`: arquivo que define os IDs de todas as classes de pedestres CityPersons e fornece mapeamento entre várias propriedades de classe.
 - `setup.py`: execute `CYTHONIZE_EVAL= python setup.py build_ext --inplace` para habilitar o plugin cython para uma avaliação mais rápida. Apenas testado para o Ubuntu.


## Evaluation
## Avaliação

Once you want to test your method on the test set, please run your approach on the provided test images and submit your results:
[Submission Page](www.cityscapes-dataset.com/submit)

The result format is described at the top of our evaluation scripts:
- [Pixel Level Semantic Labeling](cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py)
- [Instance Level Semantic Labeling](cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py)
- [Panoptic Semantic Labeling](cityscapesscripts/evaluation/evalPanopticSemanticLabeling.py)
- [3D Object Detection](cityscapesscripts/evaluation/evalObjectDetection3d.py)

Note that our evaluation scripts are included in the scripts folder and can be used to test your approach on the validation set. For further details regarding the submission process, please consult our website.

Quando quiser testar seu método no conjunto de teste, execute sua abordagem nas imagens de teste fornecidas e envie seus resultados:
[Página de envio](www.cityscapes-dataset.com/submit)

O formato do resultado é descrito na parte superior de nossos scripts de avaliação:
- [Rotulagem semântica de nível de pixel](cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py)
- [Rotulagem semântica de nível de instância](cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py)
- [Panoptic Semantic Labeling](cityscapesscripts/evaluation/evalPanopticSemanticLabeling.py)
- [Detecção de objeto 3D](cityscapesscripts/evaluation/evalObjectDetection3d.py)

Observe que nossos scripts de avaliação estão incluídos na pasta de scripts e podem ser usados ​​para testar sua abordagem no conjunto de validação. Para mais detalhes sobre o processo de submissão, por favor consulte o nosso website.

## License
## Licença

The dataset itself is released under custom [terms and conditions](https://www.cityscapes-dataset.com/license/).

The Cityscapes Scripts are released under MIT license as found in the [license file](LICENSE).

O próprio conjunto de dados é liberado sob [termos e condições] personalizados (https://www.cityscapes-dataset.com/license/).

Os Scripts do Cityscapes são liberados sob licença do MIT, conforme encontrado no [arquivo de licença](LICENÇA).

## Contact

Please feel free to contact us with any questions, suggestions or comments:

* Marius Cordts, Mohamed Omran
* mail@cityscapes-dataset.net
* www.cityscapes-dataset.com


## Contato

Por favor, não hesite em contactar-nos com quaisquer perguntas, sugestões ou comentários:

* Marius Cordts, Mohamed Omran
* mail@cityscapes-dataset.net
*www.cityscapes-dataset.com



