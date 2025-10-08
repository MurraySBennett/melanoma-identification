# Prior attempt
## Research Question
    Do the perceptual representations of skin lesion shape asymmetry(A), border regularity(B), and colour variance (C; ABCs) differ between human observers and computer-vision systems trained on these perceptual features? and, do the differences between these perceptual systems affect the diagnostic accuracy of melanoma identification?
## What did you do?
    For 10,000 skin lesion images sourced from the ISIC archive, we generated objective assessments of these features using a custom computer-vision algorithm. We then derived quantitative ratio-scale scores by applying the Bradley-Terry-Luce model to pairwise comparison data. We compared the rank-order correlation of the human and CVA assessments and compared three support vector machine classifiers trained with the (i) the human perceptual scores, (ii) the CVA assessments, and (iii) both.
## Findings?
    Computer-vision and human perceptual strength estimates of ABC features were positively but weakly correlated. Diagnostic performance was greater when the human and computer-vision assessments were combined in an SVM machine learning model as compared to models using only computer vision or human judgments metrics.
## Meaning and impact?
    These findings indicate that the representation of the ABC perceptual features used for melanoma diagnosis differ between human and computer vision observers and that combining these different representations leads to greater diagnostic performance than either perceptual system in isolation.


# In Paper
## RQ
    Practical - generate a continuous quantitative measure of the perceptual features used for melanoma identification: shape symmetry, border regularity, and colour variance -- that is based on the human perceptual system.
## What did you do?
    Using a pairwise comparison design with a bank of 10,000 skin lesion images, we applied the Bradley-Terry-Luce model to derive ratio-scale estimates of the perceptual strength of each feature for each image. We compared these estimates to those made by a computer-vision algorithm. We then evaluated the diagnostic performance of these feature estimates across three SVM models that were trained using (i) BTL estimates, (ii) computer vision estimates, or (iii) both BTL and CV estimates.
## Findings?
    We generated human-perception based estimates of feature strength for a bank of 10,000 skin lesion images. SVM classification performance was greatest when trained using both human and computer estimates of shape symmetry, border regularity, and colour variance.
## Meaning and impact?
    The image set and associated estimates offers a powerful practical tool for future research using skin lesion images. The 2AFC methodology is also encouraged for researchers looking to quickly and effectively collect continuous scale estimates of their existing images. We also found a human-machine collaborative benefit, where the combination of the BTL derived and computer vision estimates lead to greater diagnostic classification, highlighting the value of including human perceptual judgements and expertise when designing and implementing computer vision and machine-learning algorithms.