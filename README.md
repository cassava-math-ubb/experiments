# experiments
This repo contains our experimental approaches for the Cassava Leaf Disease Classification.

What are we trying to do atm:
  - Data Analysis: 
    - [x] similarity with some Monte Carlo simulations (https://arxiv.org/abs/1801.03924)
      ```
      Input: number of images loaded := 100, experiment sample size := 10, number of simulations := 10
      Runtime:             LPIPS              PSNR                SSIM
      Simulation #1: 0.6593322157859802, 9.47935962677002, 0.09285126626491547
      Simulation #2: 0.6850652098655701, 9.902604103088379, 0.08222384750843048
      Simulation #3: 0.6983931660652161, 9.53113079071045, 0.0811685174703598
      Simulation #4: 0.6989937424659729, 9.27914810180664, 0.09771811217069626
      Simulation #5: 0.6332558989524841, 9.030553817749023, 0.05319840461015701
      Simulation #6: 0.6239102482795715, 10.159566879272461, 0.06416226178407669
      Simulation #7: 0.6591953635215759, 9.76210880279541, 0.07119724154472351
      Simulation #8: 0.6343691945075989, 9.308271408081055, 0.07333287596702576
      Simulation #9: 0.658652126789093, 9.396651268005371, 0.09231053292751312
      Simulation #10: 0.6619075536727905, 10.314115524291992, 0.08350417762994766
      Overview:
      Metric        Value
      --------  ---------
      LPIPS     0.661307
      PSNR      9.61635
      SSIM      0.0791667
      ```
    - [ ] color space (https://www.sciencedirect.com/science/article/pii/S187770581100021X)
    - [ ] blurriness
    - [x] denoising filters
      [see results](https://i.imgur.com/xy0l6V6.png)
    - [x] background/foregroung segmentation
      [see results](https://i.imgur.com/AjT50es.png)
    
  - [ ] EfficientNet
  - [ ] ViT (?)
