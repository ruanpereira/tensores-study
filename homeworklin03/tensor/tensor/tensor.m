classdef tensor
    methods(Static)
%% Kronecker Product

% This function computes the Kronecker Product of two given matrices.
% Author: Kenneth B. dos A. Benicio <kenneth@gtel.ufc.br>
% Created: January 2022

function C = mtx_prod_kron(A,B)

    [ia,ja] = size(A);
    [ib,jb] = size(B);

    A = repelem(A,ib,jb);
    B = repmat(B,[ia ja]);
    C = A.*B;

end

%% Khatri-Rao Product

% This function computes the Khatri-Rao Product of two given matrices.
% Author: Kenneth B. dos A. Benicio <kenneth@gtel.ufc.br>
% Created: January 2022

function C = mtx_prod_kr(A,B)
    [ia,ja] = size(A);
    [ib,jb] = size(B);

    if (ja~=jb)
        disp('Invalid Matrices!')
        return;
    else
        C = zeros(ia*ib,ja);
        for j = 1:ja
            C(:,j) = tensor.mtx_prod_kron(A(:,j),B(:,j));
        end
    end
end

%% Unfolding

% This function computes the unfolding of a given tensor in its matrix.
% Author: Kenneth B. dos A. Benicio <kenneth@gtel.ufc.br>
% Created: 19/04/2022

function [A] = unfold(ten,mode)
    dim = size(ten);
    order = 1:numel(dim);
    order(mode) = [];
    order = [mode order];
    A = reshape(permute(ten,order), dim(mode), prod(dim)/dim(mode));
end

%% Folding

% This function computes the folding of a given matrix into its tensor.
% Author: Kenneth B. dos A. Benicio <kenneth@gtel.ufc.br>
% Created: 19/04/2022

function [ten] = fold(A,dim,mode)
    order = 1:numel(dim);
    order(mode) = [];
    order = [mode order];
    dim = dim(order);
    ten = reshape(A,dim);

    if mode == 1
        ten = permute(ten,order);
    else
        order = 1:numel(dim);
        for i = 2:mode
            order([i-1 i]) = order([i i-1]);
        end
        ten = permute(ten,order);
    end
end

%% Alternating least squares

% This function computes the folding of a given matrix into its tensor.
% Author: Kenneth B. dos A. Benicio <kenneth@gtel.ufc.br>
% Created: 16/04/2024

function [error,nmse] = ALS_estimation(ten_Y,R_approx,SNR)
    iter_max = 100;
    ten_noise = sqrt(1/(2*(10^(SNR/10))))*(randn(size(ten_Y)) + randn(size(ten_Y)));
    ten_Y_corrputed = ten_Y + ten_noise;
    Ahat = randn(15,R_approx) + 1i*randn(15,R_approx);
    Chat = randn(15,R_approx) + 1i*randn(15,R_approx);
    Bhat = randn(15,R_approx) + 1i*randn(15,R_approx);
    for i = 1:iter_max
        Ahat = tensor.unfold(ten_Y_corrputed,1)*pinv(tensor.mtx_prod_kr(Chat,Bhat).');
        Bhat = tensor.unfold(ten_Y_corrputed,2)*pinv(tensor.mtx_prod_kr(Chat,Ahat).');
        Chat = tensor.unfold(ten_Y_corrputed,3)*pinv(tensor.mtx_prod_kr(Bhat,Ahat).');
        error(i,1) = (norm(Ahat*(tensor.mtx_prod_kr(Chat,Bhat).') - tensor.unfold(ten_Y_corrputed,1),'fro')^2)/(norm(tensor.unfold(ten_Y_corrputed,1),'fro')^2);
    end
    nmse = error(end);
end

    end
end
