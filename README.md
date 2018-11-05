# Recommender-System
ML HW 
Given movies, users, ratings, find the latent U,V features to minimize \sum(rij-\hat{rij})^2. 

load(0.3) Use 30% data as the train set and 70% as the test set. 
m1_train(20) Use the traditional collaborative filtering approach to train the model. \hat{rij}=u_i \cdot v_j. Initiating the model by choosing U,V randomly. #of iterations=20. Return the MSE on the training set. 
m1_train(20,True) Use the current U,V saved in the class. train the model 20 more iterations. 
m2_train(20) Use this value as the new \hat{rij}. 

MSE_on_test_m1() mean square error if you are using model1. 
MSE_on_test_m2() mean square error if you are using model2.

recommendations(nitems, userId, silence=False, recall=False):
recommendations(10,1001) Recommend Top 10 movies for user 1001 and print the result. Return Top 10 in a list. 
recommendations(10,1001,True) Recommend Top 10 movies for user 1001. Return Top 10 in a list. 
recommendations(10,1001,True,True) Recommend Top 10 movies for user 1001. Return Top 10 which have ratings unequal to 0.(Actual ratings exist)

recall(10,4.0[optional]) If rating>=4.0[3.5], the user likes this movie. Let the system recommend the Top10 movies for each user. Calculate recall. Larger the recall, better are the recommendations. 

precision(10,4.0[optional]) Calculate the precision. Larger the precision, better are the recommendations. 

k_fold(r,kf,m1,iters):
r: the recommender class. kf: k m1: if m1=True, use model1. if m1=False, use model2. iters: number of iterations. However the MSE will be stored every 10 iterations. 

k_fold_newuser(r,kf,m1,iters):
Remove all the known users in the test set. Test completely on new users. 

p1,p2,p3: test cases. (Might have error. I didn't have time so basically test the model in the console!)

EXAMPLE: 
file_name = "movie_ratings.csv"
r1 = recommender(file_name, 10, 0.001). #Initiate the class by giving it a file_name,k value and alpha. k is the number of latent features. alpha is the learning rate. 
r2 = recommender(file_name, 10, 0.001)
k_fold(r1, 5, True, 4) #Use k-fold cross validation by setting k to 5. #iterations=4. 
k_fold(r2, kf, False, iters)
for i in range(5):
        err1 = []
        err2 = []
        for j in range(4):
            err1.append(r1.MSEs[(j * 10 + 10, i, 4)]) #MSEs[iter,i,k] means the MSE when k=k, # of iters=iter, on the ith part of the k splits. 
            err2.append(r2.MSEs[(j * 10 + 10, i, 4)][0])
        plt.plot([10,20,30,40], err1, 'r--', [10,20,30,40], err2, 'b--')
        plt.show()

#print 5 graphs of the relationship between # of iterations and MSE. red line is model1, blue line if model2. 
