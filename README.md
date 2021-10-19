# Classificação de Lesão de Pele usando GAN e Mecanismo de Atenção baseado em RoI

Este repositório apresenta o código fonte usado para a implementação do trabalho de mestrado, com objetivo de realizar a classificação de lesão de pele.

É apresentada uma metodologia de desenvolvimento, onde é aplicada a técnica de geração de imagens usando GAN a fim de crira imagens sintéticas que possam ser utilizadas para a correção do desbalanceamento do conjunto de imagens usado.

Além disso, é apresentado um modelo de CNN que utiliza máscara de lesão como método de atenção. Para a criação de tais máscaras de lesão, foi realizado treinamento de uma rede U-Net, capaz de gerar as máscaras necessárias.

Para a execução deste trabalho, é recomendado o uso do [Google Colab](https://colab.research.google.com/), ferramanta onde é possível a execução de código de aprendizado de máquina em nuvem, sendo disponibilizado gratuitamente placas de vídeo de alto desempenho.