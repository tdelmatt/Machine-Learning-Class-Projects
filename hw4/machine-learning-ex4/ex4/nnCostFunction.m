function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));



%size(X)
X1 = [ones(m, 1) X];
%size(X1)
%size(Theta1')

%layer one out
%each row is a training example, each column is a node output
%so row 1 is all node outputs for training example 1
%and a training example is a full set of x values with a y values
l2out = sigmoid(X1*(Theta1'));% + sigmoid(t1bias5000); 
l2out = [ones(size(l2out)(1), 1) l2out];

%layer two out
%each row is a training example, each column is a node output
%so row 1 is all node outputs for training example 1
%and a training example is a full set of x values with a y values
l3out = sigmoid(l2out * (Theta2'));% + sigmoid(t2bias5000);


%we need to recode y which is in 1-10 and not binary format
ytemp = [];
for i = 1 : length(y),
	ytrain = zeros(1,num_labels);  %this should be a row vector with 10 zeros
	ytrain(y(i)) = 1; %recode in binary
	ytemp = [ytemp; ytrain]; %add ytrain to ytemp
end;

%l3out;
%size(l3out);
%size(ytemp);	
	
%FROM HERE EITHER PASS ytemp AND LEAVE y UNALTERED OR CHANGE y TO = ytemp


%now lets implement the cost function
%where cost is
%
%1/m sum from 1 to m sum from 1 to k[ -y log(hyp(X)) - (1 - y)log(1-(hyp(X)))]

%also note there might be a more efficient vectorization of the regularization, although mathematically it ends up being the same combination of operations
%alternative would be theta * theta' which squares and also sums the rows and then all that needs to be summed is the columns.  
J = ((1/m) * sum(sum((((-ytemp) .* log(l3out)) - ((1 .- ytemp) .* log(1 .- l3out))),2),1)) + (lambda / (2 * m)) * (sum(sum((Theta1(:, 2:end) .* Theta1(:, 2:end)),1),2) + sum(sum((Theta2(:, 2:end) .* Theta2(:, 2:end)),1),2)) ;

%GRADIENT COMPUTATION


%NOTE
%capitoldelta or cdelta (I will refer to it as c delta) is a cumulative gradient/derivative for 
%each training example.  I guess each training example has a weighted effect on the total gradient
%which makes sense, since that's what a training example is; an example of the real thing.  We end up summing and taking the average 
%of the gradients computed by each training example in the total gradient.  

%that said, cdelta could be both thetas combined, because this method is supposed to work for any number 
%of hidden layers.  in order to do that, I guess you would just combine everything and you need to know
%the original dims to compound it back together

%i guess technically I didnt do forward propagation for an unlimited amount of vectors above
%i'm pretty sure you would need a loop to complete forward propagation for unknown hidden layers


%set capitoldelta = 0 for all ij
cdelta1 = zeros(size(Theta1));
cdelta2 = zeros(size(Theta2));


%using y(i) compute ldelda = a(L) - y(i) ////and note those are superscripts not exponentiation 
ldeltaL3 = l3out - ytemp;

%now we compute the deltas for the following layers
%ldeltaL1 = ThetaL1' * ldeltaL2 .* aL1 .* (1 - aL1)
%NOTE: THE NUMBERING OF SOME OF THESE TERMS IS OFF, NEED TO INVESTIGATE WHICH 
%TERMS CORRESPOND TO WHICH LAYER AND THEN RELABEL ACCORDINGLY
ldeltaL2 = (ldeltaL3 * Theta2) .* l2out .* (1 .- l2out);
ldeltaL2 = ldeltaL2(:,2:end);
%fprintf('l2out');
%size(l2out);



%what is the purpose of the loop if theta isn't changing
%I don't think theta can change unless if delta changes
%for i = 1 : m,
	%im pretty sure this is vectorized so no need for the loop, look into this!
	
	%we do need to compute derivatives for all the thetas so there is reasoning for 
	%computing ldeltaL1
	
	%what does iterating through the loop do though, if its not changing thetas it looks like
	%its just doing a running sum
	
	%THERE IS SOME ACCOUNTING FOR BIAS UNITS THAT NEEDS TO BE DONE HERE
	cdelta2 = cdelta2 + (ldeltaL3' * l2out);
	cdelta1 = cdelta1 + (ldeltaL2' * X1);
	
	%if J != 0,
		Theta2_grad(:, 2:end) = (1/m * cdelta2(:, 2:end)) + ((lambda/m) * Theta2(:, 2:end));
		Theta1_grad(:, 2:end) = (1/m * cdelta1(:, 2:end)) + ((lambda/m) * Theta1(:, 2:end));
	%else,
		Theta2_grad(:, 1) = (1/m * cdelta2(:, 1));
		Theta1_grad(:, 1) = (1/m * cdelta1(:, 1));
	%end;
	
	

	
		
%end; 
	%perform forward propagation to obtain a values for last layer (which we already did in order to compute the cost)
	%l2out is the cost here





% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
