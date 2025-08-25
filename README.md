# video-highlight-extraction

### 1. Introduction
본 프로젝트는 스포츠(축구) 경기 영상에서 비디오 하이라이트를 생성하는 모델을 개발합니다. 베이스 모델은 Vanderplaetes and Dupont (2020) 논문을 따릅니다.
This project aims to generate video highlight from given video and audio of Soccer game. Base model is from Action-spotting model following [Vanderplaetes and Dupont (2020)](https://arxiv.org/abs/2011.04258) 
 and [its implementation](https://github.com/bastienvanderplaetse/SoccerNetMultimodalActionSpotting). 
 
본 프로젝트는 (1) 경기 장면을 포착하는 것을 넘어 **하이라이트를 생성**하는 모델을 다룹니다. 또한 (2) GMU, CMGA, 트랜스포머 같은 fusion 전략을 실험했습니다. 
Major adaptations are as follows. (1) This project extends model's task from soloely spotting the action to **generating video highlights.** (2) It also **explored different fusion methods** including GMU, CMGA and transformer to effective process video and audion fusion. 

### 2. Data
모든 데이터셋은 피처가 추출된 .npy 파일을 이용했으며 새로운 인풋의 경우 다음과 같이 전처리 합니다. 
All train, test, validation dataset are given as feature-extracted .npy files.
For new input, data of each modality goes through the following procedure.

 ```
Video -> ResNet152 -> TruncatedSVD -> compress feature dimension to 512

 Audio -> VGGish -> PCA -> expand feature dimension to 512
```


### 3. Architecture
베이스 모델은 최종 FC층 전에 오디오와 비디오 인풋을 단순 병렬 배치해 두 모달리티를 퓨전합니다. 
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
