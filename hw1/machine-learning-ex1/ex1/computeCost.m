function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;


%MY PSEUDOCODE	
%transtheta = transpose of theta
transtheta = theta';

%multiply theta transpose by X
r1result =  X * theta;
A = r1result - y;
J = 1/(2*m) * (A' * A);
%for each element in the resulting matrix, (note : it will be a one row matrix, so only need to iterate through rows)
	%subtract each result from corresponding result in y
	%square difference
	%add to running sum
	
%multiply sum by 1/2m
%return



% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.





% =========================================================================

end
