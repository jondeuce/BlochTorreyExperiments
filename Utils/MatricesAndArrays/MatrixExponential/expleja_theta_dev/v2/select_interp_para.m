function [nsteps, gamma2, xi, dd, A, mu, c, newt, m]=...
    select_interp_para(h, A, v, extreigs, tol, m_max, p, storage,...
    humpt,proterror)

n=length(v); 

mu=(extreigs.SR+extreigs.LR)/2+1i*(extreigs.SI+extreigs.LI)/2;
isreal=true;
if (abs(extreigs.LR-extreigs.SR)/2-abs(extreigs.LI-extreigs.SI)/2)>=-10^-8
    if proterror
        newt=@newton_proterror;
    else
        newt=@newton;
    end
else
    isreal=false;
    if proterror
        newt=@newtons_proterror;
    else
        newt=@newtons;
    end
end

%Rename A to save storage
A=A-mu*speye(n,n);
[normA,c]=normAmp(1,tol(4));

sampletol=[2^-10,2^-24,2^-53];
tol_strings={'half', 'single','double'};
t=min(tol(tol(1:2)~=0)); if isempty(t) || t<2^(-53), t=2^(-53); end

index=find(sampletol<=t,1,'first');
if isempty(index), index=length(sampletol); end
if isreal
    filename=sprintf('data_leja_%s.mat',tol_strings{index});
else
    filename=sprintf('data_lejas_%s.mat',tol_strings{index});
    m_max=round(m_max/2);
end
if storage
    file=matfile(filename); else file=load(filename);
end
xi=file.xi;
theta=file.theta;
reduction=false;
if humpt>0
    dest=inf*ones(humpt,1); dest(1)=normA; j=2;
    [dest(j),k]=normAmp(j,tol(4)); c=c+k;
    while j+1<=humpt && dest(j-1)/dest(j)>1.01
        reduction=true;
        j=j+1; [dest(j),k]=normAmp(j,tol(4)); c=c+k; 
    end
end

%Gershgorin estimate for the capacity
if p<2
    if p==0
        gamma2=normA;
    else
        gamma2=sqrt((extreigs.LR-real(mu)+e)^2+(extreigs.LI-imag(mu)+e)^2);
    end
    
    mm=min(m_max,length(theta));
    [nsteps,m]=min(ceil((h*gamma2)./theta(1:mm)));
    k=m;
else
    e=file.ell_eps;
    haxis=file.haxis;
    a=h*(extreigs.LR-real(mu));
    b=h*(extreigs.LI-imag(mu));
    S=zeros(length(haxis),3);
    if reduction
       for j=1:length(haxis)
           l=ceil(sqrt([(a+e)^2,(b+e)^2]*haxis{j}(:,1)));
           S(j,:)=[l,1,l*j];
       end
    else
        for j=1:length(haxis)
            [l,o]=min(ceil(sqrt([(a+e)^2,(b+e)^2]*(haxis{j}))));
            S(j,:)=[l,o,l*j];
        end
    end
    [~,j]=min(S(:,3));
    nsteps=S(j,1);
    m=j;
    if S(j,2)>1
        k=S(j,2)+j-2;
    else
        tmp=1./sqrt(haxis{j}(1));
        k=find(theta>=tmp(1),1,'first');
    end
end
if reduction
    gamma2=h/nsteps*min(dest);
    k=find(theta>=gamma2,1,'first');
end    
gamma2=theta(k);
dd = file.dd(:,k);
if ~isreal
    m=2*m;
end
xi=xi*(gamma2/2);

    function [c, mv]=normAmp(m,p)
        switch p
            case 1,
                if m==1, c=norm(A,1); mv=0;
                else [c,mv]=normAm(A,m); c=c^(1/m); end
            case 2, if length(A)>50, [c,mv]=normest2(A,m,2,4); else c=norm(full(A)^m,2)^(1/m); mv=0; end
            case inf,
                if m==1, c=norm(A,inf); mv=0;
                else [c,mv]=normAm(conj(A),m); c=c^(1/m); end
        end
    end
end
