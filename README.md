# Liver Lesion Diagnosis Challenge on Multi-phase MRI (LLD-MMRI2023).   
<img src="https://github.com/LMMMEng/LLD-MMRI2023/blob/main/assets/img.png" width="600"/><br/>

## 🆕 **News**
* 2023-4-28: Code and Training/Validation Dataset Release.

  * We’ve enabled access to the baseline code [here](https://github.com/LMMMEng/LLD-MMRI2023/tree/main/main) and released training/validation dataset via email.
Participants who successfully registered as of April 28th have received emails with data download link , if not, please contact us with your team name at **lld_mmri@yeah.net**.
  
  * Participants who registered after April 28th will receive an email with data link within three working days.

* 2023-4-14: Registration Channel Now Open for Participants

   * Registration channel for the upcoming LLD-MMRI2023 is now open! Please complete the **[registration form](https://forms.gle/TaULgdBM7HKtbfJ97)** and you will receive an email from LLD-MMRI2023 group within 3 days. We look forward to welcoming you there!


## :dart: **Objective**     
Liver cancer is a severe disease that poses a significant threat to global human health. To enhance the accuracy of liver lesion diagnosis, multi-phase contrast-enhanced magnetic resonance imaging (MRI) has emerged as a promising tool. In this context, we aim to initiate the inaugural Liver Lesion Diagnosis Challenge on Multi-phase MRI (LLD-MMRI2023) to encourage the development and advancement of computer-aided diagnosis (CAD) systems in this domain.
## :memo: **Registration (In Progress)**   
Registration is currently underway. We kindly request participants to accurately and thoroughly complete the **[registration form](https://forms.gle/TaULgdBM7HKtbfJ97)**. The registration outcome will be communicated via email within 3 days. Please check for spam if you do not receive a reply for a long time.    

## :file_folder: **Dataset**   
**Note: The dataset is restricted to research use only, you can not use this data for commercial purposes.**
1. The training set, validation set, and test set will be made available in three stages. The training dataset (with annotations) and the validation dataset (without annotations) will be released first on **April 28th**. Annotations on the validation set will be accessible on **July 8th**, and the test dataset (without annotations) will be released on **August 2nd**.     
2. The datasets include full volume data, lesion bounding boxes, and pre-cropped lesion patches.   
3. The dataset comprises 7 different lesion types, including 4 benign types (Hepatic hemangioma, Hepatic abscess, Hepatic cysts, and Focal nodular hyperplasia) and 3 malignant types (Intrahepatic cholangiocarcinoma, Liver metastases, and Hepatocellular carcinoma). Participants are required to make a diagnosis of the type of liver lesion in each case.     
4. Each lesion has 8 different phases, providing diverse visual cues.
5. The dataset proportion is as follows：   
Training cases: 316   
Validation cases: 78   
Test cases: 104   

- [X] April 28th: Release the training set (with annotations), validation set (without annotations), and baseline code
- [ ] July 8th: Release the annotations of the validation set
- [ ] August 2nd: Release the test set (without annotations)

## 🖥️ **Training**      
We shall provide the code for data loading, model training/evaluation, and prediction in this repository. Participants can design and evaluate their models following the provided baseline. The code will be published on **April 28th**.     
**Note**: Additional public datasets are allowed for model training and/or pre-training, but private data is not allowed.  

## 🖥️ **Prediction**    
We highly suggest using our provided code to generate predictions.    
**Note**: If participants intend to use their prediction style, please ensure that the format of the prediction results is exactly the same as the template we provide.       
The code will be published on **April 28th**.   

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

## **🔍 Verification**    
To ensure fairness in the challenge, we do not allow to use private data. You have to ensure the reproducibility of the model. The top 10 teams will be required to disclose their codes and model weights on GitHub or other publicly accessible websites for verification. We shall use these codes and model weights to verify that the reproduced results are consistent with the submitted predictions. Failure to disclose codes and model weights within the stipulated time frame shall lead to removal from the leaderboard. In case of serious discrepancies detected using disclosed codes and model weights, we shall notify the corresponding teams to take remedial actions. Failure to comply within the allotted time will lead to removal from the leaderboard, and the leaderboard will be adjusted accordingly. New teams that subsequently enter the top 10 will also be required to comply with the same rules. If you use additional public datasets, you also need to disclose them; however, such disclosure will not impact your ranking.   

## 🏅 **Announcement**  
1. All the results shall be publicly displayed on the leaderboard.
2. The top 5 teams on the leaderboard shall be invited to give a 5-10 minute presentation for the MICCAI2023 challenge session. 
3. The prizes are as follows:    
   :1st_place_medal: First prize: **US$3,000** for one winner;     
   :2nd_place_medal: Second prize: **US$1,000** for one winner;    
   :3rd_place_medal: Third prize: **US$500** for two to three winners.   
 
## **🤝 Acknowledgement**  
We would like to acknowledge the following organizations for their support in making this challenge possible:    
Ningbo Medical Center Lihuili Hospital for providing the dataset.    
Deepwise Healthcare and The University of Hong Kong for organizing and providing funding for the challenge.  
## :e-mail: **Contact**  
Should you have any questions, please feel free to contact the organizers at **lld_mmri@yeah.net** or open an [issue](https://github.com/LMMMEng/LLD-MMRI2023/issues).
