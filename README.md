# onestep-extraction 
In this work, we have "extracted" training images from several diffusion models, similar to [1]. These are generated images which are exact copies of training set ones. Our attack is more efficient than [1], and our labeling can extract images which are not exactly the same, but vary in fixed spatial locations (see [below](https://github.com/ryanwebster90/onestep-extraction/tree/main#template-verbatims) ). Read about it on arxiv [A Reproducible Extraction of Training Images from
Diffusion Models](https://arxiv.org/abs/2305.08694)

To verify our attack you'll have to first generate some images, then download the corresponding images from LAION-2B, and our set of templates / masks, then verify theiry MSE is indeed low enough (or by inspection). The below code will verify our whitebox attack on SDV1:

```bash
pip install -r requirements.txt
sh verify_sdv1_wb_attack.sh 
```

# roadmap
- [x] Verify our ground truth
- [ ] Perform our whitebox and blackbox attack vs. SDV1, SDV2, DeepIF, etc.
- [ ] Verify with retrieval / template creation

[1] [Extracting training data from diffusion models. arXiv preprint arXiv:2301.13188, 2023](https://arxiv.org/abs/2301.13188)
[2] [A Reproducible Extraction of Training Images from
Diffusion Models](https://arxiv.org/abs/2305.08694)
# Some regurgitated prompts
Regurgitated prompts can be found in the following parquets. They will be labeled as 'MV','RV' or 'TV' in the 'overfit_type' field:

[deep image floyd](https://huggingface.co/datasets/fraisdufour/templates-verbs/resolve/main/groundtruth_parquets/deep_if_sdv1_wb_groundtruth.parquet)

[midjourney v4](https://huggingface.co/datasets/fraisdufour/templates-verbs/resolve/main/groundtruth_parquets/midjourney_groundtruth.parquet)

[SDV1 blackbox](https://huggingface.co/datasets/fraisdufour/templates-verbs/resolve/main/groundtruth_parquets/sdv1_bb_edge_groundtruth.parquet)

[SDV2 blackbox](https://huggingface.co/datasets/fraisdufour/templates-verbs/resolve/main/groundtruth_parquets/sdv2_bb_edge_groundtruth.parquet)

[realistic vision](https://huggingface.co/datasets/fraisdufour/templates-verbs/resolve/main/groundtruth_parquets/realistic_vision_sdv1_edge_groundtruth.parquet)



Midjourney examples
| prompt | type |
| -------- | --------|
| Prince Reunites With Warner Brothers, Plans New Album --seed 2 --stylize 0 --stop 10  | Exact   |
| Will Ferrell, John C. Reilly in Talks for <i>Border Guards</i> --seed 0 --stylize 0 --stop 10   | Exact   |
| Design Art Light in Dense Fall Forest Landscape Oversized Circle Wall Art --seed 4 --q 2    | Template  |
| Shaw Floors Spice It Up Tyler Taupe 00103_E9013 --seed 16 --stylize 0 --stop 10   | Template   |

Stable Diffusion V1
| prompt | type |
| -------- | --------|
| Galaxy Women's Leather Boots | Template   |
|Rambo 5 und Rocky Spin-Off - Sylvester Stallone gibt Updates   | Exact   |
| Full body U-Zip main opening - Full body U-Zip main opening on front of bag for easy unloading when you get to camp    | Exact    |
| Mothers influence on her young hippo   | Exact    |

Deep Image Floyd
| prompt | type |
| -------- | --------|
| Designart Green Abstract Metal Grill ContemporaryArt On Canvas - 5 Panels | Template   |
| New York Luggage Cover - New York / S - Luggage covers  | Template  |
| Foyer painted in HABANERO   | Template  |
| Shaw Floors Value Collections Sandy Hollow Cl II Net Alpine Fern 00305_5E510  | Template   |

Stable Diffusion V2
| prompt | type |
| -------- | --------|
| Pencil pleat curtains in collection Velvet, fabric: 704-18  | Template   |
| Skull Of A Skeleton With A Burning Cigarette - Vincent Van Gogh Wall Tapestry  | Template   |
| Shaw Floors Couture' Collection Ultimate Expression 15′ Sahara 00205_19829   |Template   |
| Sting Like A Bee By Louisa  - Throw Pillow   | Template |


# some other files

[Top 30K scores for the whitebox attack](https://huggingface.co/datasets/fraisdufour/sd-stuff/resolve/main/membership_attack_top30k.parquet)

[The prompts for the 2M most duplicated images](https://huggingface.co/datasets/fraisdufour/sd-stuff/resolve/main/most_duplicated_metadata.parquet)


# Template Verbatims

![templates_fig](https://github.com/ryanwebster90/onestep-extraction/assets/15658951/73ff9bdb-018b-4c12-9480-61f90e156584)

Template verbatims for various networks: Left is generated, middle is retrieved
image and right is the extracted mask. Template verbatims originate from images that have
variation in fixed spatial locations in L2B. For instance, in the top-left, varying the carpet
color in an e-commerce image. These images are generated in a many-to-many fashion (for
instance, the same prompt will generate the topleft and bottom right images, which come
from the "Shaw floors" prompts)


# Idea behind attack

![attack_model](https://github.com/ryanwebster90/onestep-extraction/assets/15658951/417e3ecd-b120-46bf-b930-e1019605f7d8)

Training images can be extracted from Stable-Diffusion in one step. In the first
row, a verbatim copy is synthesized from the caption corresponding to the image on the
second to last column. In the second row, we present verbatim copies that are harder
to detect: template verbatims. They typically represent many-to-many mappings (many
captions synthesize many verbatim templates) and thus the ground truth is constructed
with retrieval (right most column). Non-verbatims have no match, even when retrieving over the entire dataset.

Our attack exploits this fast appearance, by seperating the realistic images in the first two columns from the blurry one in the last column.

