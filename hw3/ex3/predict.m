function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

%t1nobias = Theta1(:, 1:400);
%t1bias = Theta1(:, 401);

%t1bias5000 = t1bias;
 
%for i = 1 : 4999,
%	t1bias5000 = [t1bias5000 + t1bias];
%end;

%t1bias5000 = t1bias5000';

%t2nobias = Theta2(:, 1:25);

%t2bias = Theta2(:, 26);

%t2bias5000 = t2bias;
 
%for i = 1 : 4999,
%%	t2bias5000 = [t2bias5000 + t2bias];
%end;

%t2bias5000 = t2bias5000';

%X is 5000x400 Theta1 is 25x401


X1 = [ones(m, 1) X];
%size(X1)
%size(Theta1')

l1out = sigmoid(X1*(Theta1'));% + sigmoid(t1bias5000); 
l1out = [ones(size(l1out)(1), 1) l1out];
%size(l1out)
l2out = sigmoid(l1out * (Theta2'));% + sigmoid(t2bias5000);
%size(l2out)
l2out = l2out';
[vals p] = max(l2out);
%size(p)
p = p';


% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%









% =========================================================================


end
