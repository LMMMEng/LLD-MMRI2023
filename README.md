# Liver Lesion Diagnosis Challenge on Multi-phase MRI (LLD-MMRI2023).   
<img src="https://github.com/LMMMEng/LLD-MMRI2023/blob/main/assets/img.png" width="600"/><br/>
## :dart: **Objective**     
Liver cancer is a severe disease that poses a significant threat to global human health. To enhance the accuracy of liver lesion diagnosis, multi-phase contrast-enhanced magnetic resonance imaging (MRI) has emerged as a promising tool. In this context, we aim to initiate the inaugural Liver Lesion Diagnosis Challenge on Multi-phase MRI (LLD-MMRI2023) to encourage the development and advancement of computer-aided diagnosis (CAD) systems in this domain.
## :memo: **Registration (In Progress)**   
Registration is currently underway. We kindly request participants to accurately and thoroughly complete the form available at **[:link:](https://forms.gle/TaULgdBM7HKtbfJ97)**. The review outcome will be communicated via email within a period of 10 days.    
## :file_folder: **Dataset (Coming Soon)**   
1. The training set, validation set, and test set will be made available in three stages. The training dataset (with annotations) and the validation dataset (without annotations) will be released first on **April 28th**. Annotations on the validation set will be accessible on **July 8th**, and the test dataset (without annotations) will be released on **August 2nd**.     
2. The datasets include full volume data, lesion bounding boxes, and pre-cropped lesion patches.   
3. The dataset comprises 7 different lesion types, including 4 benign types (Hepatic hemangioma, Hepatic abscess, Hepatic cysts, and Focal nodular hyperplasia) and 3 malignant types (Intrahepatic cholangiocarcinoma, Liver metastases, and Hepatocellular carcinoma). Participants are required to make a diagnosis of the type of liver lesion in each case.     
4. Each lesion has 8 different phases, providing diverse visual cues.
5. The dataset proportion is as follows：   
Training cases: 316   
Validation cases: 78   
Test cases: 104   

- [ ] April 28th: Release the training set (with annotations), validation set (without annotations), and baseline code
- [ ] July 8th: Release the annotations of the validation set
- [ ] August 2nd: Release the test set (without annotations)

## 🖥️ **Training**      
We shall provide the code for data loading, model training/evaluation, and prediction in this repository. Participants can design and evaluate their models following the provided baseline. The code will be published on **April 28**.    
## 🖥️ **Prediction**    
We highly suggest using our provided code to generate predictions.    
**Note**: If participants intend to use their prediction style, please ensure that the format of the prediction results is exactly the same as the template we provide.       
The code will be published on **April 28**.   
## **📤 Submission**     
Participants should send their prediction results to the designated email address (the email address will be notified by email to the registered participants), and we shall acknowledge receipt.   
**Note**: The challenge comprises four evaluation stages. In the first three stages, we shall update the ranking based on the predicted results of the algorithm on the validation set. Participants will have a 24-hour submission window on **May 26th**, **June 16th**, and **July 7th** to submit their prediction results. On July 8th, annotations on the validation set will be released to further support model design and training. In the final stage, the test set (without annotations) will be released on **August 2nd**. Participants are required to submit their predictions on **August 4th**, and the predicted results at this stage will determine the final ranking.      

- [ ] May 26th: The first submission of the predicted results on the validation set
- [ ] June 16th: The second submission of the predicted results on the validation set
- [ ] July 7th: The third submission of the predicted results on the validation set
- [ ] August 4th: The submission of the predicted results on the test set (this will be used for the final leaderboard).     

## :trophy: **Leaderboard**    
We shall present and update the leaderboard on our website.    
The ranking shall be determined by the average of the **F1-score** and **Cohen's Kappa coefficient**.    
## **🔍 Verificaion**    
To ensure fairness in the challenge, the top 10 teams will be required to disclose their codes and model weights on GitHub or other publicly accessible websites for verification. We shall use these codes and model weights to verify that the reproduced results are consistent with the submitted predictions. Failure to disclose codes and model weights within the stipulated time frame shall lead to removal from the leaderboard. In case of serious discrepancies detected using disclosed codes and model weights, we shall notify the corresponding teams to take remedial actions. Failure to comply within the allotted time will lead to removal from the leaderboard, and the leaderboard will be adjusted accordingly. New teams that subsequently enter the top 10 will also be required to comply with the same rules.          
## 🏅 **Announcement**  
1. All the results shall be publicly displayed on the leaderboard.
2. The top 5 teams on the leaderboard shall be invited to give a 5-10 minute presentation for the MICCAI2023 challenge session. 
3. The prizes are as follows:    
   :1st_place_medal: First prize: :dollar:3,000 for one winner;     
   :2nd_place_medal: Second prize: :dollar:1,000 for one winner;    
   :3rd_place_medal: Third prize: :dollar:500 for one to three winners.   
 
## **🤝 Acknowledgement**  
We would like to acknowledge the following organizations for their support in making this challenge possible:    
Ningbo Medical Center Lihuili Hospital for providing the dataset.    
Deepwise Healthcare and The University of Hong Kong for organizing and providing funding for the challenge.  
## :e-mail: **Contact**  
Should you have any questions, please feel free to contact the organizers at **lld_mmri@yeah.net** or open an [issue](https://github.com/LMMMEng/LLD-MMRI2023/issues).
