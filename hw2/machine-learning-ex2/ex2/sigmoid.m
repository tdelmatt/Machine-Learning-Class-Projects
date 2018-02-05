function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


B = ones(size(z));
C = ones(size(z));
B = B * e;
B = B .^ (-z);
B = B + 1;

g = C ./ B;


% =============================================================

end
