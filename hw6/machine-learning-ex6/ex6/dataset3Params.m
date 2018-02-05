function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.



%{

C = 1;
sigma = 0.3;

values = [.01 .03 .1 .3 1 3 10 30];
%prediction = zeros(8,8);
minerror = [999999999 0 0];
for i = 1: 8,
	for j = 1:8,
		
		%train data for c = i, sigma = j
		model = svmTrain(X, y, values(i), @(x1, x2) gaussianKernel(x1, x2, values(j))); 
		
		%predict
		predictions = svmPredict(model, Xval);
		temperror = mean(double(predictions ~= yval))
		
		%if new error is less, replace current min error
		if temperror < minerror(1),
			fprintf('NEW MINIMUM ERROR!')
			minerror = [temperror i j]
		end;
		
	end;
end;

%C = minerror(2);
%sigma = minerror(3);


%}

C = 1;
sigma = .1;


%get indices of minimum error
%set C and sigma to those values


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
