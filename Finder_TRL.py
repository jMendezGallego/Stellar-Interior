# Importamos bibliotecas

import numpy as np

###########################################################
###########################################################

# Función para el cálculo de la producción de energía

def energia(T,P,X,Y):
    # Cálculo de la densidad
    Z=1-X-Y
    mu = 1/(2*X+0.75*Y+0.5*Z)
    H = 1/6.02214076e+23    # [g]
    boltz = 1.380649e-23    # [J/K]
    rho = mu*H/boltz*(P*10e8)/(T*10e7)
    
    
    # Rangos de temperatura
    # Cadena pp
    if T < 0.4:
        nu_pp = 0
        e_pp = 0
    elif T < 0.6:
        nu_pp = 6
        e_pp = 10**-6.84
    elif T < 0.95:
        nu_pp = 5
        e_pp = 10**-6.04
    elif T < 1.2:
        nu_pp = 4.5
        e_pp = 10**-5.56
    elif T < 1.65:
        nu_pp = 4
        e_pp = 10**-5.02
    elif T < 2.4:
        nu_pp = 3.5
        e_pp = 10**-4.40
    else:
        nu_pp = 0
        e_pp = 0
    
    # Ciclo CN
    if T < 1.2:
        nu_cn = 0
        e_cn = 0
    elif T < 1.6:
        nu_cn = 20
        e_cn = 10**-22.2
    elif T < 2.25:
        nu_cn = 18
        e_cn = 10**-19.8
    elif T < 2.75:
        nu_cn = 16
        e_cn = 10**-17.1
    elif T < 3.6:
        nu_cn = 15
        e_cn = 10**-15.6
    elif T < 5.0:
        nu_cn = 13
        e_cn = 10**-12.5
    else:
        nu_cn = 0
        e_cn= 0
        
    Epp = e_pp*(X**2)*rho*((T*10)**nu_pp)
    Ecn = e_cn*X*Z/3*rho*((T*10)**nu_cn)
    cual = (Ecn < Epp)      # Qué reacción domina?
    if cual == True:
        reac = 'pp'
        nu = nu_pp
        e = e_pp
    else:
        reac = 'CN'
        nu = nu_cn
        e = e_cn
    if e_pp==0:        # Se produce energía?
        reac = '--'
    
    return (reac,nu,e)

###########################################################
###########################################################

# Función para el cálculo de capas exteriores

def fase_rad(X,Y,M_tot,R_tot,L_in):
    
    # Inicialización de las variables necesarias

    R_in=0.9*R_tot
    Z = 1-X-Y       # Metalicidad
    if Z < 0.02:    # Vijilamos la opacidad en función de la metalizidad
        print('Error a la hora de caluclar la opacidad. La absorción bf no es la dominante ya que Z < 2%.')
        print('Domina absorción ff.')

    mu = 1/(2*X+0.75*Y+0.5*Z)   # Peso mulecular medio cte.

    n_capa = 101   # Número de capas en las que se va a dividir el radio

    # Inicializamos los vectores de las variables para caso radiat.

    r_rad = np.linspace(R_in,0,n_capa)
    M_rad = np.zeros(n_capa)
    P_rad = np.zeros(n_capa)
    L_rad = np.zeros(n_capa)
    T_rad = np.zeros(n_capa)

    N = np.zeros(n_capa)

    # Inicializamos las f_i para método Predictor-Corrector
    fm = np.zeros(n_capa)
    fp = np.zeros(n_capa)
    fl = np.zeros(n_capa)
    ft = np.zeros(n_capa)

    #############################################################

    # INTEGRACIÓN DESDE LA SUPERFICIE

    #############################################################

    # CAPAS INICIALES. 3 capas superficiales
    # L_in=cte. (No hay creación de energía)
    # M_tot=cte. (Densidad superficial muy baja)

    h = r_rad[0]-r_rad[1]   # Paso en r

    for i in range(3):      # Cálculo de las tres capas más superficiales
        M_rad[i] = M_tot
        L_rad[i] = L_in
        T_rad[i] = (1.9022*mu*M_tot)*(1/r_rad[i]-1/R_tot)      # Se usa R_tot
        P_rad[i] = (10.645*np.sqrt(M_tot/(mu*Z*(1+X)*L_in)))*T_rad[i]**4.25
        
        # f para método diferencias (fm=fl=0 -> ctes). Caso radiativo
        ft[i] = -(0.01679*Z*(1+X)*mu**2)*(P_rad[i]**2)*L_rad[i]/((T_rad[i]**8.5)*r_rad[i]**2)
        fp[i] = -(8.084*mu)*P_rad[i]*M_tot/(T_rad[i]*r_rad[i]**2)

    #############################################################
    
    # FASE RADIATIVA. Integración desde superficie

    i=2     # Desde la tercera capa
    cota_err = 0.0001

    N[i] = T_rad[i]*fp[i]/(P_rad[i]*ft[i])      # Calculamos n+1 para las primeras capas
    while N[i] > 2.5:      # Fase radiativa
            
        # Cálculo de las Delta_P[i] y Delta_T[i]
        DP1 = -h*fp[i]+h*fp[i-1]
        DP2 = -h*fp[i]+2*h*fp[i-1]-h*fp[i-2]
        DT1 = -h*ft[i]+h*ft[i-1]    # h es negativo
        # Cálculo P_est y T_est
        P_est = P_rad[i]-h*fp[i]+(DP1/2)+(5/12*DP2)
        T_est = T_rad[i]-h*ft[i]+DT1/2
        P_old = P_rad[i]
        T_old = T_rad[i]
        
        while abs((T_est-T_old)/T_est) > cota_err:
            
            while abs((P_est-P_old)/P_est) > cota_err:
                # Masa calculada
                fm[i+1] = 0.01523*mu*P_est*(r_rad[i+1]**2)/T_est
                DM1 = -h*fm[i+1]+h*fm[i]    # Delta_M[i+1]
                M_rad[i+1] = M_rad[i]-h*fm[i+1]-DM1/2
                
                #Presión calculada a partir de la anterior masa
                fp[i+1] = -(8.084*mu)*P_est/T_est*M_rad[i+1]/(r_rad[i+1]**2)
                DP1 = -h*fp[i+1]+h*fp[i]    # Delta_P[i+1]
                P_rad[i+1] = P_rad[i]-h*fp[i+1]-DP1/2
                
                P_old = P_est
                P_est = P_rad[i+1]
            
            # Producción energía
            t_reac,nu,e = energia(T_est,P_rad[i+1],X,Y)    # Factores producción energía
            if t_reac == 'pp':
                fl[i+1] = 0.01845*e*(X**2)*(10**nu)*(mu**2)*(P_rad[i+1]**2)*(T_est**(nu-2))*r_rad[i+1]**2        
            elif t_reac =='CN':
                fl[i+1] = 0.01845*e*X*Z/3*(10**nu)*(mu**2)*(P_rad[i+1]**2)*(T_est**(nu-2))*r_rad[i+1]**2
            else:
                fl[i+1] = 0
            
            # Luminosidad calculada
            DL1 = -h*fl[i+1]+h*fl[i]
            DL2 = -h*fl[i+1]+2*h*fl[i]-h*fl[i-1]
            L_rad[i+1] = L_rad[i]-h*fl[i+1]-(DL1/2)-(DL2/12)
            
            # Temperatura calculada a partir de todo lo demás
            ft[i+1] = -0.01679*Z*(1+X)*mu**2*(P_rad[i+1]**2)*(L_rad[i+1])/((T_est**8.5)*r_rad[i+1]**2)
            DT1 = -h*ft[i+1]+h*ft[i]
            T_rad[i+1] = T_rad[i]-h*ft[i+1]-DT1/2
            
            T_old = T_est
            T_est = T_rad[i+1]
            
        i+=1
        if i > (n_capa-1):      # Para poner un límite al bucle while
            print('Todas las capas son radiativas.')
            break
        
        N[i] = T_rad[i]*fp[i]/(P_rad[i]*ft[i])
        
        if N[i] <= 2.5:
            frontera = i    # Marcador de frontera radiativa

    # Este bucle recoge una capa que no es radiativa.

    #############################################################

    # INTERPOLACIÓN. Parte radiativa
    # Acortamos los vectores
    r_rad = np.flip(r_rad[frontera-1:frontera+1])
    M_rad = np.flip(M_rad[frontera-1:frontera+1])
    P_rad = np.flip(P_rad[frontera-1:frontera+1])
    L_rad = np.flip(L_rad[frontera-1:frontera+1])
    T_rad = np.flip(T_rad[frontera-1:frontera+1])
    N_rad = np.flip(N[frontera-1:frontera+1])


    # Interpolamos valores de los parámetros para la capa n+1=2.5
    r_rad_f = np.interp(2.5,N_rad,r_rad)
    M_rad_f = np.interp(2.5,N_rad,M_rad)
    P_rad_f = np.interp(2.5,N_rad,P_rad)
    L_rad_f = np.interp(2.5,N_rad,L_rad)
    T_rad_f = np.interp(2.5,N_rad,T_rad)
    
    return (h,frontera,r_rad_f,M_rad_f,P_rad_f,L_rad_f,T_rad_f)

##################################################################
##################################################################

# Función para calcular capas interiores y error relativo en frontera.

def fase_conv(X,Y,h,frontera,r_rad_f,M_rad_f,P_rad_f,L_rad_f,T_rad_f,Tc_in):
    
    # Valores necesarios para el tratamiento de la fase convectiva.
    Z=1-X-Y
    mu = 1/(2*X+0.75*Y+0.5*Z)
    K = P_rad_f/(T_rad_f**2.5)  # Se utilizan los valores estimados en la 1ª capa convectiva calculada de froma radiativa
    cota_err = 0.0001
    
    # Inicializamos los vectores para la parte convectiva
    r_conv = np.arange(0,r_rad_f+h,h)
    front_int = len(r_conv)-1
    M_conv = np.zeros(front_int+1)
    P_conv = np.zeros(front_int+1)
    L_conv = np.zeros(front_int+1)
    T_conv = np.zeros(front_int+1)

    # Inicializamos las f_i para método Predictor-Corrector
    fm = np.zeros(front_int+1)
    fl = np.zeros(front_int+1)
    ft = np.zeros(front_int+1)

    #############################################################

    # CAPAS INICIALES. Fase convectiva con M y L =0

    for i in range(3):      # Tres capas centrales
        # Cálculo de los parámetros
        M_conv[i] = 0.005077*mu*K*(Tc_in**1.5)*r_conv[i]**3
        T_conv[i] = Tc_in-0.008207*(mu**2)*K*(Tc_in**1.5)*r_conv[i]**2
        P_conv[i] = K*(T_conv[i]**2.5)
        
        # Produción energía
        t_reac,nu,e = energia(Tc_in,P_conv[0],X,Y)
        if t_reac == 'pp':
            L_conv[i] = 0.00615*e*X**2*(10**nu)*mu**2*K**2*(Tc_in**(3+nu))*r_conv[i]**3
        elif t_reac == 'CN':
            L_conv[i] = 0.00615*e*X*Z/3*(10**nu)*mu**2*K**2*(Tc_in**(3+nu))*r_conv[i]**3
        else:
            if i == 0:
                L_conv[0] = 0
            else:
                L_conv[i] = L_conv[i-1]
                
        # f para método diferencias
        fm[i] = 0.01523*mu*K*(T_conv[i]**1.5)*(r_conv[i]**2)   
        if r_conv[i] != 0:    
            ft[i] = 3.234*mu*M_conv[i]/(r_conv[i]**2)     # en r=0 no se puede evaluar f[t]
        if t_reac == 'pp':
            fl[i] = 0.01845*e*X*X*(10**nu)*mu**2*K**2*(T_conv[i]**(nu+3))*r_conv[i]**2
        elif t_reac =='CN':
            fl[i] = 0.01845*e*X*Z/3*(10**nu)*mu**2*K**2*(T_conv[i]**(nu+3))*r_conv[i]**2
        else:
            fl[i] = 0
        
    #############################################################
        
    # FASE CONVECTIVA. Integración desde el centro.

    for i in range(2,front_int):
        
        # Estimación de temperatura.
        DT1 = h*ft[i]-h*ft[i-1]
        T_est = T_conv[i]+h*ft[i]+DT1/2
        T_old = T_conv[i]
        
        while abs((T_est-T_old)/T_est) > cota_err:
            # Presión estimada (polítropo) y masa calculada
            P_est = K*(T_est**2.5)
            fm[i+1] = 0.01523*mu*P_est*r_conv[i+1]**2/T_est
            DM1 = h*fm[i+1]-h*fm[i]
            M_conv[i+1] = M_conv[i]+h*fm[i+1]-DM1/2
            
            # Cálculo de la nueva temperatura
            ft[i+1] = -3.234*mu*M_conv[i+1]/r_conv[i+1]**2
            DT1 = h*ft[i+1]-h*ft[i]
            T_conv[i+1] = T_conv[i]+h*ft[i+1]-DT1/2
            
            T_old = T_est 
            T_est = T_conv[i+1]
        
        P_conv[i+1] = K*(T_conv[i+1]**2.5)
        # Producción energía
        t_reac,nu,e = energia(T_conv[i+1],P_conv[i+1],X,Y)
        if t_reac == 'pp':
            fl[i+1] = 0.01845*e*X*X*(10**nu)*mu**2*(P_conv[i+1]**2)*(T_conv[i+1]**(nu-2))*r_conv[i+1]**2
        elif t_reac == 'CN':
            fl[i+1] = 0.01845*e*X*Z/3*(10**nu)*mu**2*(P_conv[i+1]**2)*(T_conv[i+1]**(nu-2))*r_conv[i+1]**2
        else:
            fl[i+1] = 0
        
        # Cálculo de la luminosidad
        DL1 = h*fl[i+1]-h*fl[i]
        DL2 = h*fl[i+1]-2*h*fl[i]+h*fl[i-1]
        L_conv[i+1] = L_conv[i]+h*fl[i+1]-(DL1/2)-(DL2/12)
            
    ############################################################

    # INTERPOLACIÓN. Fase convectiva
    # Acortamos los vectores
    r_conv = r_conv[front_int-1:front_int+1]
    M_conv = M_conv[front_int-1:front_int+1]
    P_conv = P_conv[front_int-1:front_int+1]
    L_conv = L_conv[front_int-1:front_int+1]
    T_conv = T_conv[front_int-1:front_int+1]


    # Interpolamos valores de los parámetros para la capa n+1=2.5
    M_conv_f = np.interp(r_rad_f,r_conv,M_conv)
    P_conv_f = np.interp(r_rad_f,r_conv,P_conv)
    L_conv_f = np.interp(r_rad_f,r_conv,L_conv)
    T_conv_f = np.interp(r_rad_f,r_conv,T_conv)

    #############################################################

    # AJUSTE A RADIO INTERMEDIO
    # Cálculo del error relativo.
    relative_err = np.sqrt(((M_conv_f-M_rad_f)/M_rad_f)**2+((P_conv_f-P_rad_f)/P_rad_f)**2+((L_conv_f-L_rad_f)/L_rad_f)**2+((T_conv_f-T_rad_f)/T_rad_f)**2)

    return relative_err

#########################################################################
#########################################################################

# BUSCADOR DE LA TEMPERATURA CENTRAL

def find_Tc(X,Y,M_tot,R_tot,L_in,Tc_in):
    
    # Cálculo de las capas exteriores
    (h,frontera,r,M,P,L,T) = fase_rad(X,Y,M_tot,R_tot,L_in)
    
    # Con los datos calculamos la parte central variando Tc
    Tc = np.arange(Tc_in-1.5,Tc_in+1.5,0.01)
    Err = np.ones(len(Tc))
    
    for i in range(len(Tc)):
        Err[i] = fase_conv(X,Y,h,frontera,r,M,P,L,T,Tc[i])

    i_min = np.where(Err == min(Err))[0]
    
    Tc = np.arange(Tc[i_min-1],Tc[i_min+1],abs(Tc[i_min-1]-Tc[i_min+1])/1000)
    Err = np.ones(len(Tc))

    for i in range(len(Tc)):
        Err[i] = fase_conv(X,Y,h,frontera,r,M,P,L,T,Tc[i])

    i_min = np.where(Err == min(Err))[0]
    
    Tc = Tc[i_min]
    Err_min = Err[i_min]
    
    return (Tc,Err_min)

##########################################################################
##########################################################################

def find_TRL(X,Y,M_tot,R_tot,L_in,Tc_in):
    # Creamos un array para almacenar los Errores
    # L eje y // R eje x

    errores = np.ones([11,11])   # Dimensiones tienen que ser impar
    temp_central = errores*0
    Error_min = np.zeros(np.size(errores,0))
    j_min = np.zeros(np.size(errores,0))

    delta_L = 5
    delta_R = 0.2

    L = np.arange(L_in-(np.size(errores,0)-1)/2*delta_L,L_in+(np.size(errores,0)-1)/2*delta_L+0.1,delta_L)
    R = np.arange(R_tot-(np.size(errores,1)-1)/2*delta_R,R_tot+(np.size(errores,1)-1)/2*delta_R+0.1,delta_R)

    for i in range(np.size(errores,0)):     # Para cada L
        for j in range(np.size(errores,1)):     # Para cada R
            
            temp_central[i,j], errores[i,j] = find_Tc(X,Y,M_tot,R[j],L[i],Tc_in)
        

    # Buscamos el minimo error

    for i in range(np.size(errores,0)):
        Error_min[i] = min(errores[i,:])
        j_min[i] = np.where(errores[i,:]==Error_min[i])[0]


    I = np.where(Error_min==min(Error_min))[0]
    J = int(j_min[I])

    L_in = L[I]
    R_tot = R[J]

    delta_L = 2*delta_L/np.size(errores,0)
    delta_R = 2*delta_R/np.size(errores,1)

    L = np.arange(L_in-(np.size(errores,0)-1)/2*delta_L,L_in+(np.size(errores,0)-1)/2*delta_L+0.1,delta_L)
    R = np.arange(R_tot-(np.size(errores,1)-1)/2*delta_R,R_tot+(np.size(errores,1)-1)/2*delta_R+0.1,delta_R)

    for i in range(np.size(errores,0)):     # Para cada L
        for j in range(np.size(errores,1)):     # Para cada R
            
            temp_central[i,j], errores[i,j] = find_Tc(X,Y,M_tot,R[j],L[i],Tc_in)
        

    # Buscamos el minimo error

    for i in range(np.size(errores,0)):
        Error_min[i] = min(errores[i,:])
        j_min[i] = np.where(errores[i,:]==Error_min[i])[0]


    I = np.where(Error_min==min(Error_min))[0]
    J = int(j_min[I])
    
    
    
    
    
    
    delta_L = 2*delta_L/np.size(errores,0)
    delta_R = 2*delta_R/np.size(errores,1)

    L = np.arange(L_in-(np.size(errores,0)-1)/2*delta_L,L_in+(np.size(errores,0)-1)/2*delta_L+0.1,delta_L)
    R = np.arange(R_tot-(np.size(errores,1)-1)/2*delta_R,R_tot+(np.size(errores,1)-1)/2*delta_R+0.1,delta_R)

    for i in range(np.size(errores,0)):     # Para cada L
        for j in range(np.size(errores,1)):     # Para cada R
            
            temp_central[i,j], errores[i,j] = find_Tc(X,Y,M_tot,R[j],L[i],Tc_in)
        

    # Buscamos el minimo error

    for i in range(np.size(errores,0)):
        Error_min[i] = min(errores[i,:])
        j_min[i] = np.where(errores[i,:]==Error_min[i])[0]


    I = np.where(Error_min==min(Error_min))[0]
    J = int(j_min[I])
    
    
    
    
    
    
    

    Error_min = errores[I,J]
    Tc = temp_central[I,J]

    L_in = L[I]
    R_tot = R[J]
    
    return (Error_min,Tc,L_in,R_tot)

##########################################################################
##########################################################################

# Parámetros inicio

# Definimos los parámetros de le estrella
# Parámetros constantes

X = 0.75
Y = 0.22
M_tot = 8.41

# Valores iniciales (varían cuando resolvemos la estrella)

R_tot = 16
L_in = 1100

# Valor inicial de la temperatura central
T0 = 2

Error_min,Tc,Lum,R = find_TRL(X,Y,M_tot,R_tot,L_in,T0)
