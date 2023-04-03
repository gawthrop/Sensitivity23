import control as con
import numpy as np
import matplotlib.pyplot as plt

def Sloppy(Sys,t=None,GainOnly=False,small = 1e-12):
    
    # Dimensions
    D = Sys.D
    n_out = D.shape[0]
    n_par = D.shape[1]

    # Hessian matrix H
    ## Start with small diagonal matix to avoid singularity
    H = small*np.identity(n_par)

    if GainOnly:
        # Compute Steady state (DC) gain
        G = con.dcgain(Sys)

        # Compute Hessian
        H += G.T@G

    else:
       # Compute unit step response
       Step = con.step_response(Sys,T=t)
    
       # Extract data
       t = Step.time
       dt = t[1]-t[0]
       y = Step.y
       
       ## Compute Hessian
       for yi in y:
           #print(yi.shape)
           H += (yi@yi.T)
           
       ## Normalise
       ## H *= (dt/max(t))
       H /= len(t)
        
    ## Normalise H wrt number of outputs and parameters
    ##H /= n_out*n_par
    
    ## Eigen analysis
    eig,eigv = np.linalg.eig(H)
        
    ## Remove spurious imag parts
    eig = np.real(eig)
    eigv = np.real(eigv)

    ## Sort eigenvalues with largest first
    sort_perm = np.flip(eig.argsort())
    eig = eig[sort_perm]
    eigv = eigv[:, sort_perm]
        
                  
    return H,eig,eigv,t

def SloppyPrint(eig,eigv,inp,min_eig = 1e-6,min_eigv=1e-6,max_eigs=5,
                prefix=r'\lambda_',precision = 3):
    
      for i,eigi in enumerate(eig):
        line = ''
        if (eigi/eig[0] > min_eig) and (i<max_eigs):
            # print(f'Eigenvalue: {eigi:.4f}')
            line += f'\sqrt\sigma_{i+1} &= {np.sqrt(eigi):.2g} & V_{i+1}\Lambda &='
            eigvi = eigv[:,i]
            #print(eigvi)
            #print('norm',np.sum(eigvi*eigvi))

            maxeig = max(abs(eigvi))
           # print(maxeig)
            np.set_printoptions(precision=precision)
            #print(f'Eigenvector: {eigvi/maxeig}')

            ## Find most important parameters
            i_sort = np.flip(np.argsort(abs(eigvi)))
            #print(i_sort)
            #print(inp)
            #inpSort = []
            if eigvi[i_sort[0]] < 0:
                # Change sign
                eigvi *= -1
                
                
            for i in i_sort:
                #inpSort.append(inp[i])
                eigvii = eigvi[i]
                
                if abs(eigvii) > min_eigv:
                    if (eigvii<0):
                        sign = '-'
                    else:
                        sign = '+'
                    inpi = inp[i]
                    inpi = inpi[1:] #Strip leading s
                    if not prefix is None:
                        inpi = '{'+inpi+'}'
                    line += f' {sign} {abs(eigvii):.3f} {prefix}{inpi}'
            
            #print(inpSort)
            #print(eigvi)
            print(line)
    
def SloppyPlot(eig,eigv,inp,square=False,Eig=None,Eigv=None,
               grid=False,lw=4,ls='dashed',color='black'):

    if not (len(eig)==2):
        print('Plotting only for 2 parameters, not', len(eig))
    else:
        ## Plot Elipse
        theta = np.linspace(0, 2*np.pi, 1000);
        ellipsis = (1/np.sqrt(eig[None,:]) * eigv) @ [np.sin(theta), np.cos(theta)]
        plt.plot(ellipsis[0,:], ellipsis[1,:],label='$Q=1$',lw=lw,color=color)
        if grid:
            plt.grid()

        xlabel = inp[0]
        xlabel = '$\lambda_{'+xlabel[1:]+'}$'
        plt.xlabel(xlabel)

        ylabel = inp[1]
        ylabel = '$\lambda_{'+ylabel[1:]+'}$'
        plt.ylabel(ylabel)
            
        ## Save these axes
        xmin,xmax = plt.xlim()
        ymin,ymax = plt.ylim()

        ## Optional square axes
        if square:
            xymin = min(xmin,ymin)
            xymax = max(xmax,ymax)

        ## Optional plot second elipse - eg for steady-state version
        if not (Eig is None):
            ellipsis = (1/np.sqrt(Eig[None,:]) * Eigv) @ [np.sin(theta), np.cos(theta)]
            plt.plot(ellipsis[0,:], ellipsis[1,:],label='$Q_\infty=1$',
                     lw=lw,color=color,ls=ls)
            plt.legend()
        ## Use first set of axes
        if square:
            plt.xlim(left=xymin,right=xymax)
            plt.ylim(bottom=xymin,top=xymax)
        else:
            plt.xlim(left=xmin,right=xmax)
            plt.ylim(bottom=ymin,top=ymax)

        
       # plt.show()

         

    ## Plot eigenvalues
    # #log10eig = np.flip(np.sort(np.log10(eig)))
    # sortEig = np.flip(np.sort(eig))
    # log10sortEig = np.log10(sortEig)
    # plt.bar(range(len(eig)),sortEig)
    # plt.show()
    

def SloppyPlotData(t,y,inp,outp,grid=False,lw=4):
    
    n_out = y.shape[0]
    n_par = y.shape[1]
    ## Plot Data
    for i in range(n_out):
        for j in range(n_par):
            plt.plot(t,y[i,j,:],label=f'{inp[j]}-{outp[i]}',lw=lw)
    plt.legend()
    if grid:
        plt.grid()
    plt.xlabel(r'$t$')
    plt.ylabel(r'f')
    plt.show()
               
def sloppy(Sys,inp,outp,t=None):
    sys = extractSubsystem(Sys,sc,sf,inp,outp)
    print(sys)
    H,eig,eigv,t,y = Sloppy(sys,t=t)
    SloppyPlot(eig,eigv,inp)
    SloppyPlotData(t,y,inp,outp)
    
