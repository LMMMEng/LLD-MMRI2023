# Liver Lesion Classification Challenge on Multi-phase MRI (LLD-MMRI2023).   
<img src="https://github.com/LMMMEng/LLD-MMRI2023/blob/main/images/logo.png" width="600"/><br/>
## **Purpose**   
Liver cancer remains one of the most severe diseases threatening human health globally, while multi-phase contrast-enhanced magnetic resonance imaging (MRI) can deliver more accurate liver lesion diagnosis. Therefore, we plan to host the first Liver Lesion Diagnosis Challenge on Multi-phase MRI (termed LLD-MMRI2023) to promote the research of such CAD systems.
## **Registration**   
Please carefully fill out the [form](example.com) and send it to xxx@xxx.com, we will reply to the review results by email within 10 days.  
## **Dataset**   
1. Dataset can be downloaded at: [Google Drive](example.com), [BaiduNetDisk](example.com), training and validation sets will be released on Apr 28, and the test set will be released on May 30.      
2. The datasets includes the acquired full volume data with lesion bounding box and our pre-cropped lesion patch.   
3. There is 7 different lesion types including including 4 benign types (Hepatic hemangioma, Hepatic abscess, Hepatic cysts, and Focal nodular hyperplasia) and 3 malignant types (Intrahepatic cholangiocarcinoma, Liver metastases, and Hepatocellular carcinoma).   
4. Each lesion has 8 different phases for providing diverse visual clues.   
5. Dataset proportion：   
Training cases: 316   
Validation cases: 78   
Test cases: 104   

## **Training**      
We will provide code for data loading, model training and evaluation, and prediction in this repository. You can follow the provided baseline to design and evaluate your model.   
The code will be published on Apr 28.
## **Prediction**    
It is suggested to use our provided code to generate predictions on the test set.  
**Note**: If you would like to employ your own prediction style, please make sure that the format of the prediction results is exactly the same as the template we provided.    
The code will be published on Apr 28.
## **Submission**     
Please sent your prediction file to xxx@xxx.com, we will get back to you when we receive the results.   
**Note**: In each test stage, if a team submits multiple times, we will only use the first submission.
## **Leaderboard**    
We will present and update the leaderboard here.    
The ranking was determined by the average of the F1-score and Cohen's Kappa coefficient.    
## **Announcement**  
1. All the results will be publicly displayed on the leaderboard.   
2. The top-5 teams on board will be invited to give a 5-10 minute presentation for the MICCAI2023 challenge session.   
## **Acknowledgement**  
**Ningbo Medical Center Lihuili Hospital** provided the dataset.    
**Deepwise Healthcare** and **The University of Hong Kong** will provide funding for the challenge. Many thanks!
