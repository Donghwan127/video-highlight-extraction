# video-highlight-extraction
### 1. Introduction

This project aims to generate video highlight from given video and audio of Soccer game. Base model is from Action-spotting model following [Vanderplaetes and Dupont (2020)](https://arxiv.org/abs/2011.04258) 
 and [its implementation](https://github.com/bastienvanderplaetse/SoccerNetMultimodalActionSpotting). 
 
Major adaptations are as follows. (1) This project extends model's task from soloely spotting the action to **generating video highlights.** (2) It also **explored different fusion methods** including GMU, CMGA and transformer to effective process video and audion fusion. 

### 2. Data

All train, test, validation dataset are given as feature-extracted .npy files.
For new input, data of each modality goes through the following procedure.

 ```
Video -> ResNet152 -> TruncatedSVD -> compress feature dimension to 512

 Audio -> VGGish -> PCA -> expand feature dimension to 512
```


### 3. Architecture

The base model from Vanderplaetes and Dupon (2020) employs early fusion by concatenating audio and video input right before the final FC. Refer to *Figure 1. Method 4*. 


 
#### Encoder
Main difference of our model from the base model is the method of fusion. We have explored different fusion methods from (1) GMU(Gated Multi-Modal Unit) to (2) CMGA(Cross-Modality Gated Attention) and (3) Transformer instead of simple concatenation. 

#### Decoder (Inference) 
With the given logits from encoder, decoder spot peaks and span spotted peaks with pre-defined offsets to generate Highlights

<Our model architecture 삽입>

### 4. Usage

```bash
git clone https://github.com/Donghwan127/video-highlight-extraction
cd #repo
pip install -r requirements.txt

# train 
python ClassificationMinuteBased.py --architecture CMGAArchi2 --training listgame_Train_300.npy --validation listgame_Valid_100.npy --testing listgame_Test_100.npy --featuresVideo ResNET --featuresAudio VGGish --PCA --network VLAD --tflog Model --VLAD_k 128--WindowSize 20 --outputPrefix vlad-**cmgaarchi2-20sec  --formatdataset 1
**best model 및 설정 확인 필요**
```
```
# inference
bash run_inference.sh
```

### 5. References
[Vaderplaestse and Dupont (2020)](https://arxiv.org/abs/2011.04258). Improved Soccer Action Spotting using both Audio and Video Streams

[Jiang and Ji (2022)](https://arxiv.org/abs/2208.11893). Cross-Modality Gated Attention Fusion for Multimodal Sentiment Analysis. 

[John et al. (2017)](https://arxiv.org/abs/2208.11893).  Gated Multimodal Units for Information Fusion.

[Giancola et al. (2018)](https://arxiv.org/abs/1804.04527). SoccerNet: A Scalable Dataset for Action Spottin in Soccer Videos.

---
### 7. Contributors

**Donghwan Seo** | github 

Adapted base model source codes for highlight-generating task including adjusting class label and class weight while training 

Ran experiments using different architectures, losses and weight control


**Haejin Cho** | github

Implemented and integrated Cross-Modality Gated Attention into the architecture with different moment of fusoin시기 !! 

Implemented and integrated Attention Bottleneck into the architecture

**Haeun Jeon** | github

Implemented and integrated Gated Modality Unit into the architecture

implemented inference codes for highlight video generation.

Ran experiments using different architectures, losses and weight control

---
This project is part of [KUBIG](https://www.kubigkorea.com/)  2025 FALL contest. 
