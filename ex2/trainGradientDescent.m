function [theta_final,J]= trainGradientDescent(X,y,theta,num_iter,lambda)
  theta_final=zeros(size(X,2),1);
  J=0;
  m=size(X,1);
  for i=1:num_iter
    grads= X'*(sigmoid(X*theta)-y);
    theta=theta - (lambda/m)*(grads);
  end
  theta_final=theta;
   I= ones(m,1);
  cost= -(I-y).*log(I-sigmoid(X*theta)) - y.*log(sigmoid(X*theta)) ;
  
  J= sum(cost)/(m);  
 