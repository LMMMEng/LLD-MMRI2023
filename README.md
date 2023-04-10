# Liver Lesion Classification Challenge on Multi-phase MRI (LLD-MMRI2023).   
<img src="https://github.com/LMMMEng/LLD-MMRI2023/blob/main/images/logo.png" width="600"/><br/>
## **Purpose**     
Liver cancer remains one of the most severe diseases threatening human health globally, while multi-phase contrast-enhanced magnetic resonance imaging (MRI) can deliver more accurate liver lesion diagnosis. Therefore, we plan to host the first Liver Lesion Diagnosis Challenge on Multi-phase MRI (termed LLD-MMRI2023) to promote the research of such CAD systems.
## **Registration (In Progress)**   
Please carefully fill out the :point_right:**[form](https://forms.gle/TaULgdBM7HKtbfJ97)**, we will reply to the review results by email within 10 days.  
## **Dataset (Coming Soon)**   
1. Dataset can be downloaded at: The download link will be presented on **April 28th**.    
The training set, validation set, and test set will be released sequentially in three stages. First, the training dataset (with annotations) and the validation dataset (without annotations) will be released on April 28th. Second, annotations on the validation set will be accessible on July 8th. Third, the test dataset (without annotations) will be released on August 2nd.     
2. The datasets includes the acquired full volume data with lesion bounding box and our pre-cropped lesion patch.   
3. There is 7 different lesion types including including 4 benign types (Hepatic hemangioma, Hepatic abscess, Hepatic cysts, and Focal nodular hyperplasia) and 3 malignant types (Intrahepatic cholangiocarcinoma, Liver metastases, and Hepatocellular carcinoma).   
4. Each lesion has 8 different phases for providing diverse visual clues.   
5. Dataset proportion：   
Training cases: 316   
Validation cases: 78   
Test cases: 104   
## **Training**      
We will provide code for data loading, model training and evaluation, and prediction in this repository. You can follow the provided baseline to design and evaluate your model.   
The code will be published on April 28.
## **Prediction**    
It is suggested to use our provided code to generate predictions.  
**Note**: If you would like to employ your own prediction style, please make sure that the format of the prediction results is exactly the same as the template we provided.    
The code will be published on April 28.   
## **Submission**     
Please sent your prediction file to xxx@xxx.com, we will get back to you when we receive the results.   
**Note**: The challenge consists of four evaluation stages. In the first three stages, the ranking will be updated based on the predicted results of the algorithm on the validation set. Participants will have a 24-hour submission window on May 26th, June 16th, and July 7th to submit their prediction results. On July 8th, annotations on the validation set will be released to further support model design and training. In the final stage, the test set (without annotations) will be released on August 2nd. Participants are required to submit their predictions on August 4th, and the predicted results at this stage will determine the final ranking.     
## **Leaderboard**    
We will present and update the leaderboard here.    
The ranking was determined by the average of the F1-score and Cohen's Kappa coefficient.    
## **Verificaion**    
To ensure the fairness of the challenge, the top-10 teams will be required to disclose their codes and model weights on GitHub or other publicly accessible websites for verification. We will use these codes and model weights to verify that the reproduced results are consistent with the submitted predictions. If teams fail to disclose their codes and model weights within the stipulated time frame, they will be removed from the leaderboard. If serious discrepancies are detected using disclosed codes and model weights, we will notify the corresponding teams to take remedial actions. If they fail to complete within the allotted time, they will be removed from the leaderboard and the leaderboard will be adjusted accordingly. New teams that subsequently enter the top-10 will also be required to comply with the same rules.     
## **Announcement**  
1. All the results will be publicly displayed on the leaderboard.   
2. The top-5 teams on board will be invited to give a 5-10 minute presentation for the MICCAI2023 challenge session.   
## **Acknowledgement**  
**Ningbo Medical Center Lihuili Hospital** provided the dataset.    
**Deepwise Healthcare** and **The University of Hong Kong** will provide funding for the challenge. Many thanks!
