# Food Segmentation

## Meta Segment Anything Models (SAM)

| Model | Type |
| :----: | :----: |
| [sam_vit_h_4b8939.pth](https://drive.google.com/file/d/1kJ4xaiVVTpUXEeF9phvkhilxsbpToHW4/view?usp=share_link) | vit_h |
| [sam_vit_l_0b3195.pth](https://drive.google.com/file/d/1VBVipk1izHDOHH7zWKMHGhzfQv3FeP0v/view?usp=share_link) | vit_l |
| [sam_vit_b_01ec64.pth](https://drive.google.com/file/d/12seklpq0KeMd3uWCTiGeODBrrqy5bod1/view?usp=share_link) | vit_b |

### Amazon Mechanical Turk

* [Official Website](https://requester.mturk.com/)
* [Suggested Price](https://aws.amazon.com/sagemaker/data-labeling/pricing/)
    * Only have 102 ingredient categories + 1 background + 1 other ingredient now, may need to add more categories.
        * But we don't know what other categories we will need, we only know when we are labelling the image, it will increase the difficulty of assigning tasks on Turk.
            * If the food ingredient does not belong to the current 102 categories, it may only be possible for the person who is labeling the food to assign these ingredients to "other ingredient", and then we have to further process the image in "other ingredient" ourselves.
    * The performance of the SAM Demo provided by Meta is much better than that of their open source models.
        * [It has not been solved yet](https://github.com/facebookresearch/segment-anything/issues/54), we may need to manually adjust the parameters. It is challenge due to the black-box nature of model hyperparameters.
    * Estimated Price: $0.05 * (400 each category) * (104 categories) * (3 triple budget) * (1.35 SGD to USD) = S$8424
        * The actual price will likely be higher than the estimated price, it depends on the performance of the SAM model.
