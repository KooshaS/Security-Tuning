clc, clear all


mu = [0 0];
Sigma = [1 0; 0 1];
x1 = -5:.2:5; x2 = -5:.2:5;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mu,Sigma);
F = reshape(F,length(x2),length(x1));

mvncdf([0 0],[1 1],mu,Sigma);
contour(x1,x2,F,[.0001 .001 .01 .05:.1:.95 .99 .999 .9999]);
xlabel('x'); ylabel('y');
line([0 0 1 1 0],[1 0 0 1 1],'linestyle','--','color','k');

[F,err] = mvncdf([-2 -2],[2 2],mu,Sigma)