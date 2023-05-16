# onestep-extraction
In this work, we have "extracted" training images from several diffusion models, similar to [1]. These are generated images which are exact copies of training set ones. Our attack is more efficient than [1], and our labeling can extract images which are not exactly the same, but vary in fixed spatial locations (see below). Read it on arxiv soon.

To verify our attack you'll have to first generate some images, then download the corresponding images from LAION-2B, and our set of templates / masks, then verify theiry MSE is indeed low enough (or by inspection). The below code will verify our whitebox attack on SDV1:

```bash
pip install -r requirements.txt
sh 
```



[1] Extracting training data from diffusion models. arXiv preprint arXiv:2301.13188, 2023.

# roadmap

- [x] Verify our ground truth
- [ ] Perform our whitebox and blackbox attack vs. SDV1, SDV2, DeepIF, etc.
- [ ] Verify with retrieval / template creation
- [ ] 

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

