# MedChat

Objective: To create a chatbot that shall provide people with knowledge about genetic disorders, genetic testing and how to utilize the testing kits.
Based on symptoms and family history determine which disease is probable and which ones could be transferred to the next generation.
Hence, assisting the general public in understanding the relationships between diseases and genes in order to improve general public’s medical knowledge and understanding
Innovations: It's providing a very refined version of answer available from the net as it has been trained on Pegasus and BERT model after which the data is being displayed to the user when asking questions on the chatbot. It consists of valuable and summarized information.
This is important as it shall raise awareness amongst the people about genetic disorder because around 10% of the people around the world have genetic disorders and they get to know in their adulthood.The chatbot shall help people recognize if they have slightest of the symptoms for the genetic disorder cause many people fear early doctor visits so after relating to data on bot they can refer the doctor for proper medication and prevent from the disorder to reach an alarming stage in case it's dangerous to health.

What dataset tested: Articles from Pubmed were tested for the following 5 diseases: ehlers danlos syndrome, charcot marie tooth, von willebrand's disease, tourette syndrome, ankylosing spondylitis and genetic testing kits. 
BBC dataset was used to train the BERT model and  SQUAD dataset to train QnA set.
Got the dataset from articles available at https://pubmed.ncbi.nlm.nih.gov/ , since medical database isn’t available due to privacy issues and hence articles were utilized. 
Analysis: For the analysis of the model, scorer has been used for plotting BERT model (scoring the different tokens assigned to the various terms) then for testing the chatbot several time to find the accuracy.
Algorithms used: K means clustering for topic modelling, HuggingFace transformer library for the BERT model 
Visualization: Tabular presentation of the result of K-means Clustering, visual plotter for different scores of the tokens generated in BERT model and the chatbot is live presentation of working of model.
System modules: Pytorch, Transformers, BERT, Google Pegasus XSUM model,MS Virtual Agent Chatbot, Azure cloud services
