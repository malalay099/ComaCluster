#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Required libraries

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import scipy.stats as st
import astropy.units as u
from scipy.stats import norm
from PyAstronomy import pyasl
from astropy import units as u
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.coordinates import SkyCoord
from scipy.stats import norm, gaussian_kde, maxwell


# In[2]:


# We load the data.

df_coma = pd.read_csv('E:\comas\coma clusters.csv')

#The head function in Python displays the first five rows of the dataframe by default.
df_coma


# In[3]:


#c  = 3e+5*(u.km/u.s)

# Velocity of the galaxy
#v_gxy = velocity.to_numpy()*(u.km/u.s)
#size_v = np.size(v_gxy)

#slist = []
#for i in v_gxy:
    #print(i)
 #   j = i.value / c.value 
    #print(j)
 #   slist.append(j)
#z = np.array(slist) 
#unt = i.unit / c.unit

#print(z*unt)


# In[4]:


df_coma = df_coma.dropna(subset=['Velocity  '])


# In[5]:


df_coma.columns


# In[6]:


velocity = df_coma['Velocity  ']
magnitude =  df_coma['Magnitude   ']
sns.distplot(df_coma['Velocity  '], kde=True, bins=200, color = 'b')
plt.xlabel('velocity (kms$^{-1}$)')
plt.ylabel('Count')
fig, ax = plt.subplots()
ax = velocity.hist(bins=100,density=True, color='b')
ax.set_xlim([2500,11000])
#plt.title('نمودار سرعت_قدر ظاهری')
plt.xlabel('velocity (kms$^{-1}$)')
plt.ylabel(' Magnitude ')
plt.show()


# In[7]:


# محور افقی فراوانی(قدر ظاهری) و محور قایم سرعت کهکشان هاست.
magnitude
fig2, ax2 = plt.subplots()
ax2 = plt.scatter(magnitude,velocity, marker="*", color='b', label = 'Data1+Data2')
plt.xlim([10,25])
plt.ylim([3000,12000])
#plt.savefig('my.fig7.png', dpi=300)
plt.xlabel("Magnitude")
plt.ylabel("Velocity (kms$^{-1}$)")
#plt.title('subplots')
#plt.legend(loc = 'upper left')
plt.show()


# In[8]:


# Loc
# از لیبل یا برچسب سطر و ستون برای دسترسی به آن استفاده میکند.
coma = df_coma.loc[(df_coma['Velocity  '] >=2000) & (df_coma['Velocity  '] <=12000)]


# In[9]:


# هسیتوگرام رسم میکنیم

hist, bin_edge = np.histogram(coma['Velocity  '], 20, density=True)


# In[10]:


# تایع گاووسی 
# تابع اولی تایع گاوسی است 
# تابع دومی لگاریتم تابع گاوسی است.
def gauss(x,a,mu,sigma):
    y = a - mu*(x-sigma)**2
    return y
def gauss2(x,a,mu,sigma):
    y = np.exp(a-mu*(x-sigma)**2)
    return y


# In[11]:


# Len:  برای محاسبه طول لیست
n = len(bin_edge)
hist_x = np.zeros(n-1)
for i in range(n-1):
    hist_x[i] = (bin_edge[i]+bin_edge[i+1])/2
hist_y = hist


# In[12]:


hist_y = np.where(hist_y == 0, 1e-10, hist_y)
coef, cov = curve_fit(gauss, hist_x, np.log(hist_y))
#print(coef)
#print(cov)


# In[13]:


num_bins = 100

fig, ax = plt.subplots()
ax = coma['Velocity  '].hist(bins=num_bins,density=True)
#ax.plot(hist_x,gauss2(hist_x, coef[0],coef[1],coef[2]), 'r')
ax.set_xlim([4000, 10000])
#plt.xlabel('Velocity')
#plt.ylabel('Magnitude')
#plt.title('Normal Distribution')
#plt.legend()

plt.show()


# In[14]:


#‌محاسبه و رسم مو و زیگما با توجه به نمودار
velocity = velocity[(velocity >= 4000) & (velocity <= 10000)]
app_mag = coma['Magnitude   ']
mu, sigma = norm.fit(velocity)


# In[15]:


num_bins = 20
fig, ax = plt.subplots()
n, bins, patches = ax.hist(velocity, num_bins, density=True, color='b')

bincenters = 0.5*(bins[1:]+bins[:-1])
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5* (1 / sigma * (bins - mu))**2))

ax.plot(bins, y, '--', c='r')


#ax.set_xlabel('Velocity(m/s)')
#ax.set_ylabel('Magnitude')
ax.set_title('$\mu=$ %0.2f, $\sigma=$  %0.2f' % (mu, sigma))
plt.xlabel('Velocity (kms$^{-1}$)')
plt.ylabel('Magnitude ')
fig.tight_layout()
plt.show()


# In[16]:


sigma = 0.5/np.sqrt(coef[1])
# سرعت متوسط
v_mean = coef[2]
# سرعت مینیمم
v_min = v_mean - 2*sigma
# سرعت ماکزیمم
v_max = v_mean + 2*sigma
print('Mean Velocity = {0:4.2f},   Minimum Velocity = {1:4.2f},   Maximum Velocity = {2:4.2f}'.format(v_mean, v_min, v_max))


# In[17]:


# تعداد کهکشان ها ما که در زیر نمودار تابع توزیع گاووسی است مجموع اش 545 کهکشان است.
# و اینجا میانگین، ماکزیمم، مینیمم و استندرد دیویشن سرعت 545 کهکشان، قدر ظاهری 545 کهکشان و محورات اینها آن 545 کهکشان است.

coma.describe()
#coma.head()
#coma


# In[18]:


coma.columns


# In[65]:


# RA and Dec values (example data)
x = coma['RA']
y = coma['Dec']

ra = ngc['ra']
dec = ngc['de']

# Create a scatter plot
plt.scatter(x, y, color='b', marker = '*')
plt.scatter(ra, dec, color='r', marker = '*')

plt.annotate('NGC4889', xy=(195.1, 29))
plt.annotate('NGC 4874', xy=(194.87,28.9))
plt.annotate('NGC 4839', xy=(194.4,28.2))

# Set the title of the plot
plt.xlim(195.4,194)
#plt.ylim(27.6,29.5)
degree_sign = u'\N{DEGREE SIGN}'
plt.xlabel('RA [' + degree_sign + ']')
plt.ylabel('Dec [' + degree_sign + ']')

# Display the plot
plt.show()


# In[20]:


# Hubble Law (H_law = 67.4+/-0.5)
# توسط قانون هابل فاصله تک تک کهکشان زیر نمودار را بدست اوردیم

H_law = 67.4*(u.km/(u.s*u.Mpc))

# Velocity of the galaxy
v_gxy = velocity.to_numpy()*(u.km/u.s)
size_v = np.size(v_gxy)

## slist = np.empty([0],dtype=type(H_law))
slist = []
for i in v_gxy:
    #print(i)
    j = i.value / H_law.value 
    #print(j)
    slist.append(j)
d_gxy = np.array(slist) 
unt = i.unit / H_law.unit

#print("mean of distance galaxy: ", np.mean(d_gxy)) 
print(d_gxy*unt)


# In[21]:


# از قانون هابل اول فاصله ها را بدست اوردم بعد ضرب ضریب ثابت هابل کردم که اینجا ماکزیمم و مینیمم فاصله هاست.

vmax_gxy = 9889.00    # km/s
vmin_gxy = 4732.00      # km/s

# Hubble Law (67.66+42 and 67.66-42)

H_law = 67             # km/s Mpc
h100 = 0.67

dmax_gxy = vmax_gxy / H_law
dmin_gxy = vmin_gxy / H_law
d_max = dmax_gxy*h100
d_min = dmin_gxy*h100

print('maximum distance:', d_max*(u.Mpc))
print('minimum distance:', d_min*(u.Mpc))
#print(dmax_gxy)
#print(dmin_gxy)


# In[22]:


# فاصله میانگین و فاصله استندرد دیویشن را از طریق قانون هابل بدست اوردم
# اینجا فاصله میانگین یعنی فاصله هابلی کل کهکشان ها 99 میگا پارسک شد

v_mean = 6980.00
v_sigma = 1085.25
# H_law (67.66+0.42 and 67.66-0.42)
H_law = 67
h100 = 0.67

d_law = v_mean / H_law
d_sigma = v_sigma / H_law
d = d_law*h100
d_std = d_sigma*h100
print('Distance of Coma Cluster: ', d_law*(u.Mpc))
print('mean distnace: ', d*(u.Mpc))
#print('standard deviation of distance :', d_sigma*(u.Mpc))
#print('standard deviation of distance:', d_std*(u.Mpc))


# In[23]:


# سرعت شعاعی یا قطر ظاهری
# alpa 0.7 degree (0.012 radian)

α = 0.012       # unit: (u.r)
#d_law = 100     # unit: (u.Mpc)

D = α * d_law
print('apparent diameter:', D*u.Mpc)


# In[24]:


# Absolute Magnitude
# در اینجا از رابطه زیر قدر مطلق تک تک کهکشان ها بدست امد 

# am (apparent magnitudes of the galaxy )
am = app_mag.to_numpy()

# abs_m  or absMag_gxy (Absolute Magnitude  of the galaxy)

def absMag_gxy(appMag_gxy, d_law):
    absMag_gxy = appMag_gxy-5*(np.log10(d_law*1e+6/10))
    return absMag_gxy
abs_m = []
for i in range(np.size(am)):
    abs_m.append(absMag_gxy(am[i], d_law))
    
print(abs_m)


# In[25]:


# از رابطه بالا میانگین قدر مطلق، ماکزیمم و مینیمم قدر مطلق را بدست اوردم

appM_max = 22.61
appM_min = 2.20
appM_mean = 17.53
d_max = 105             # Mpc
d_min = 103             # Mpc
d_mean = 102            # Mpc

#Mu_max = 5*(np.log10(d_max*1e+6/10))
#Mu_min = 5*(np.log10(d_min*1e+6/10))
#Mu_mean = 5*(np.log10(d_law*1e+6/10))

absM_max = appM_max-35.10
absM_min = appM_min-35.06
absM_mean = appM_mean-35

print('Absolute Magnitude maximum : ', absM_max)
print('Absolute Magnitude minimum: ', absM_min)
print('Absolute Magnitude mean : ', absM_mean)


# In[26]:


# در اینجا درخشندگی تک تک کهکشان ها بدست میاریم
#Luminosity of the galaxy
Lum_sun = 3.85e+26    #*(u.watt)

def Lum_gxy (Lum_sun, absMag_gxy):
    Lum_gxy = 84*10**(-2/5*(absMag_gxy))*Lum_sun
    return Lum_gxy
lum_gxy=[]
for i in range(np.size(abs_m)):
    lum_gxy.append(Lum_gxy (Lum_sun, abs_m[i]))
    
print(lum_gxy*u.watt)


# In[27]:


#Luminosity of the galaxy
Lum_sun = 3.85e+26    #*(u.watt)

ab = -17.47
abc = -12.49
abn = -32.86
Lumgxy_max = 84*10**(-2/5*(abc))*Lum_sun                          
Lumgxy_min = 84*10**(-2/5*(abn))*Lum_sun                          
Lumgxy_mean = 84*10**(-2/5*(ab))*Lum_sun                       

print('Luminosity galaxy of maximum: ', Lumgxy_max*(u.watt))
print('Luminosity galaxy of minimum:', Lumgxy_min*(u.watt))
print('Luminosity galaxy of mean:', Lumgxy_mean*(u.watt))


# In[28]:


#  ..در اینجا از رابطه ذیل جرم تک تک کهکشان های زیر نمودار تابع گاووسی را بدست اورده و در سلول بعدی میانگین، مینیممم و ماکزیمم را بدست اوردم

# Galaxy Mass

def Mass_gxy(absMag_gxy):
    Mass_gxy = 3e+28*5e+3*10**(-2/5*(absMag_gxy))
    return Mass_gxy

# m_g (mass of the galaxy)
Mg=[]
for i in range(np.size(abs_m)):
    Mg.append(Mass_gxy(abs_m[i]))
    
print(Mg*u.kg)
#print('mean of mass galaxy: ', np.mean(m_g))


# In[29]:


# جدول زیر مقادیر 191 کهکشان(جرم، درخشندگی، قدر مطلق، قدر ظاهری و مدل فاصله) آنهاست

my_dict = {
    'Lum_gxy ' : lum_gxy,
    'Mass_gxy ' : Mg,
    'App_mag ' : am,
    'Absolute ' : abs_m
    
};

clusters = pd.DataFrame(my_dict)
clusters


# In[30]:


# جدول زیر مقادیر 191 کهکشان(جرم، درخشندگی، قدر مطلق، قدر ظاهری و مدل فاصله) آنهاست

my_dict = {
    'Distance ': d_gxy
    
};

cluster0 = pd.DataFrame(my_dict)
#cluster0
#clusters.describe()
clusters.sum()


# In[31]:


# مقدار سیگما را از روی نمودار بالا داریم
# MN_gxy (Mass.N of galaxy): جرم کل را هم از جدول بالا داریم

σ = 1e+6              # m/s
MN_gxy = 2.14e+45           # kg         #جرم کل 191 کهکشان 
K_egy = 3/2*MN_gxy*σ**2

print('Kinetic_energy of galaxy:', K_egy*u.Joule)


# In[32]:


# :میانگین انرژی جنبشی 191 کهکشان خوشه ی گیسو عبارت از

N = 545
K_mean = K_egy/N

print('The Kinetic Energy mean of galaxy:', K_mean*u.J)


# In[33]:


# انرژی پتانسیل را بدست میاریم

G = 6.6e-11              # m^3/kg*s^2
MN_gxy = 2.1e+45      # kg
R = 3.0e+22              # m

P_e = -2/5*G*(MN_gxy**2)/R
print('Potential energy:', P_e*(u.J))


# In[34]:


N = 545
P_mean = P_e/N

print(P_mean)


# In[35]:


# زمان خوشه گیسو
# اگر زمانی ازین مرتبه صبر کنیم شعاع این خوشه تقربیآ دو برابر میشود

δv = 1085.25           # km/s
R_gxy = 3.08e+19        # km
T = R_gxy/δv
print('Times of coma cluster:', T*(u.second))


# In[36]:


# جرم ویریال

σ = 1e+7          # unit: m/s
R_vir = 3.08e+22          # unit: m
G = 6.67e-11              # unit: m^3/kg.s^2

M_vir = 5*R_vir*(σ**2)/G
print('Mass Virial of galaxy:', M_vir*u.kg)


# In[37]:


# چگالی خوشه گیسو

#R = 1Mpc
#v = 2.92e+67              # m^3
V = 1e+68
ρ_mean = M_vir/V
print('Density of galaxy:', ρ_mean*(u.kg/u.m**3))


# In[38]:


# Read the data from a CSV file

coma2004 = pd.read_csv('E:\\2004.csv')
coma2004.head()


# In[39]:


# Define the center coordinates in hours, minutes, degrees and seconds
ra_center = [12, 59, 42.75]
dec_center = [28, 58, 14.4]

# Convert the center coordinates to degrees
ra_center_deg = (ra_center[0] + ra_center[1]/60 + ra_center[2]/3600) * 15
dec_center_deg = dec_center[0] + dec_center[1]/60 + dec_center[2]/3600 


# In[40]:


# Set the x and y axis coordinates in J2000 coordinates
x_coord = coma2004['X'] 
y_coord = coma2004['Y']

# Convert the RA and Dec offsets to degrees
ra_offset_deg = x_coord / 60
dec_offset_deg = y_coord / 60

# Calculate the RA and Dec coordinates for each data point
ra_deg = ra_center_deg + ra_offset_deg
dec_deg = dec_center_deg + dec_offset_deg

# Calculate the x and y coordinates in arcseconds
x_arcsec = (ra_deg - ra_center_deg) * 60 * np.cos(np.deg2rad(dec_center_deg))
y_arcsec = (dec_deg- dec_center_deg) * 60

# Determine whether the x and y coordinates are positive or negative
x_sign = np.where(x_arcsec >= 0, 1, -1)
y_sign = np.where(y_arcsec >= 0, 1, -1)

# Convert the x and y coordinates to absolute values
x_arcsec_abs = np.abs(x_arcsec)
y_arcsec_abs = np.abs(y_arcsec)

# Convert the x and y coordinates to degrees
x_deg = x_arcsec_abs / 60
y_deg = y_arcsec_abs / 60

# Determine the sign of the x and y coordinates in degrees
x_sign_deg = np.where(x_sign == 1, 'E', 'W')
y_sign_deg = np.where(y_sign == 1, 'N', 'S')


# In[41]:


# Create a numpy array with the final data
##data = np.column_stack((ra_deg, dec_deg, x_deg, y_deg, x_sign_deg, y_sign_deg))
data = np.column_stack((np.round(ra_deg,2), np.round(dec_deg,2), np.round(x_deg,2), np.round(y_deg,2), 
                        x_sign_deg, y_sign_deg))

# Save the data to a CSV file
np.savetxt('coma2004_degrees.csv', data, delimiter=',', header='RA,Dec,X,Y,X_sign,Y_sign', fmt='%s')


# In[42]:


new_coma = pd.read_csv('coma2004_degrees.csv')
new_coma.head()


# In[43]:


# RA and Dec values (example data)
ra_values = new_coma['# RA']
dec_values = new_coma['Dec']

# Create a scatter plot
plt.scatter(ra_values, dec_values, color='g', marker = '*', label = 'Data3(2004)')

# Set the x-axis label
plt.xlabel('x_axis (degrees)')
plt.ylim(28,29.5)
# Set the y-axis label
plt.ylabel('y_axis (degrees)')
plt.legend(loc='lower left')

# Display the plot
plt.show()


# In[44]:


# Read the data from a CSV file
coma_GMP = pd.read_csv('E:\\comaa.csv')
coma_GMP.head()


# In[45]:


# Define the center coordinates in hours, minutes, degrees and seconds
ra_center = [12, 59, 42.75]
dec_center = [28, 58, 14.4]

# Convert the center coordinates to degrees
ra_center_deg = (ra_center[0] + ra_center[1]/60 + ra_center[2]/3600) * 15 - 0.1
dec_center_deg = dec_center[0] + dec_center[1]/60 + dec_center[2]/3600


# In[46]:


# Set the x and y axis coordinates in J2000 coordinates
x_coord = coma_GMP['x_axis'] 
y_coord = coma_GMP['y_axis']

# Convert the RA and Dec offsets to degrees
ra_offset_deg = x_coord / 3600
dec_offset_deg = y_coord / 3600

# Calculate the RA and Dec coordinates for each data point
ra_deg = ra_center_deg + ra_offset_deg
dec_deg = dec_center_deg + dec_offset_deg

# Calculate the x and y coordinates in arcseconds
x_arcsec = (ra_deg - ra_center_deg) * 3600 * np.cos(np.deg2rad(dec_center_deg))
y_arcsec = (dec_deg- dec_center_deg) * 3600

# Determine whether the x and y coordinates are positive or negative
x_sign = np.where(x_arcsec >= 0, 1, -1)
y_sign = np.where(y_arcsec >= 0, 1, -1)

# Convert the x and y coordinates to absolute values
x_arcsec_abs = np.abs(x_arcsec)
y_arcsec_abs = np.abs(y_arcsec)

# Convert the x and y coordinates to degrees
x_deg = x_arcsec_abs / 3600
y_deg = y_arcsec_abs / 3600

# Determine the sign of the x and y coordinates in degrees
x_sign_deg = np.where(x_sign == 1, 'E', 'W')
y_sign_deg = np.where(y_sign == 1, 'N', 'S')


# In[47]:


# Create a numpy array with the final data
##data = np.column_stack((ra_deg, dec_deg, x_deg, y_deg, x_sign_deg, y_sign_deg))
data = np.column_stack((np.round(ra_deg,2), np.round(dec_deg,2), np.round(x_deg,2), np.round(y_deg,2), 
                        x_sign_deg, y_sign_deg))

# Save the data to a CSV file
np.savetxt('comaGMP_degrees1.csv', data, delimiter=',', header='RA,Dec,X,Y,X_sign,Y_sign', fmt='%s')


# In[48]:


new_comaGMP = pd.read_csv('comaGMP_degrees1.csv')
new_comaGMP


# In[49]:


# RA and Dec values (example data)
ra_values = new_comaGMP['# RA']
dec_values = new_comaGMP['Dec']

# Create a scatter plot
plt.scatter(ra_values, dec_values, color='b', marker = '*', label = 'Data1')

# Set the x-axis label

# Set the x-axis label
plt.xlabel('Right Ascension (degrees)')
# Set the y-axis label
plt.ylabel('Declination (degrees)')
plt.ylim(28,29.5)
#plt.xlim(195.7,194.4)
plt.legend(loc='lower left')
# Display the plot
plt.show()


# In[50]:


# Read the data from a CSV file
mgc = pd.read_csv('E:\\mgc.csv')
mgc.head()


# In[51]:


# Define the center coordinates in hours, minutes, degrees and seconds
ra_centers = [12, 59, 42.75]
dec_centers = [28, 58, 14.4]

# Convert the center coordinates to degrees
ra_center_deg = (ra_center[0] + ra_center[1]/60 + ra_center[2]/3600) * 15 - 0.1
dec_center_deg = dec_center[0] + dec_center[1]/60 + dec_center[2]/3600


# In[52]:


# Set the x and y axis coordinates in J2000 coordinates
x_coord = mgc['x_axis'] 
y_coord = mgc['y_axis']

# Convert the RA and Dec offsets to degrees
ra_offset_deg = x_coord / 3600
dec_offset_deg = y_coord / 3600

# Calculate the RA and Dec coordinates for each data point
ra_deg = ra_center_deg + ra_offset_deg
dec_deg = dec_center_deg + dec_offset_deg

# Calculate the x and y coordinates in arcseconds
x_arcsec = (ra_deg - ra_center_deg) * 3600 * np.cos(np.deg2rad(dec_center_deg))
y_arcsec = (dec_deg- dec_center_deg) * 3600

# Determine whether the x and y coordinates are positive or negative
x_sign = np.where(x_arcsec >= 0, 1, -1)
y_sign = np.where(y_arcsec >= 0, 1, -1)

# Convert the x and y coordinates to absolute values
x_arcsec_abs = np.abs(x_arcsec)
y_arcsec_abs = np.abs(y_arcsec)

# Convert the x and y coordinates to degrees
x_deg = x_arcsec_abs / 3600
y_deg = y_arcsec_abs / 3600

# Determine the sign of the x and y coordinates in degrees
x_sign_deg = np.where(x_sign == 1, 'E', 'W')
y_sign_deg = np.where(y_sign == 1, 'N', 'S')


# In[53]:


# Create a numpy array with the final data
##data = np.column_stack((ra_deg, dec_deg, x_deg, y_deg, x_sign_deg, y_sign_deg))
mgc = np.column_stack((np.round(ra_deg,2), np.round(dec_deg,2), np.round(x_deg,2), np.round(y_deg,2), 
                        x_sign_deg, y_sign_deg))

# Save the data to a CSV file
np.savetxt('mgcs.csv', mgc, delimiter=',', header='RA,Dec,X,Y,X_sign,Y_sign', fmt='%s')


# In[54]:


new_mgc = pd.read_csv('mgcs.csv')
new_mgc.head()


# In[55]:


ngc = pd.read_csv('E:NGC.csv')
#ngc.columns
ngc


# In[56]:


# Read the data from a CSV file

comaMb = pd.read_csv('E:\comas\\coma2001_edit.csv')
comaMb.head()


# In[57]:


# Define the columns containing the RA and Dec in hours, minutes, and seconds

ra_cols = ['RA_hours', 'RA_minutes', 'RA_seconds']
dec_cols = ['Dec_degrees', 'Dec_minutes', 'Dec_seconds']


# In[58]:


# Convert RA and Dec to degrees for each row in the DataFrame
for index, row in comaMb.iterrows():
    # Get the RA and Dec values for this row
    ra_hours, ra_minutes, ra_seconds = row[ra_cols]
    dec_degrees, dec_minutes, dec_seconds = row[dec_cols]
    
     # Convert RA to degrees
    ra_degrees = (ra_hours + ra_minutes/60 + ra_seconds/3600) * 15 
    
    # convert Dec to degrees
    dec_sign = -1 if dec_degrees < 0 else 1
    dec_degrees2 = dec_sign * (abs(dec_degrees) + dec_minutes/60 + dec_seconds/3600) +1

# Update the DataFrame with the converted values
    comaMb.at[index,  'RA_degrees'] = ra_degrees
    comaMb.at[index, 'Dec_degress'] = dec_degrees2


# In[59]:


# Save the updated DataFrame to a new CSV file
comaMb.to_csv('comaMb_degrees02.csv', index=False)


# In[60]:


#print(data)
new_comaMb = pd.read_csv('comaMb_degrees02.csv')
new_comaMb


# In[61]:


# RA and Dec values (example data)
ra_values = new_comaMb['RA_degrees']
dec_values = new_comaMb['Dec_degress']

# Create a scatter plot
fig, ax1 = plt.subplots()
ax1.scatter(ra_values, dec_values, color='b', marker = '*', label = 'Data3')

# Set the x-axis label
#plt.xlabel('R.A (degrees)')

plt.xlim(195.3,194)
#plt.ylim(28.5,26.5)

#degree_sign = u'\N{DEGREE SIGN}'
#plt.xlabel('RA [' + degree_sign + ']')
#plt.ylabel('Dec [' + degree_sign + ']')
plt.show()


# In[62]:


# RA and Dec values (example data)
x_values = new_coma['# RA']
y_values = new_coma['Dec']

# RA and Dec values (example data)
ra_values = new_comaGMP['# RA']
dec_values = new_comaGMP['Dec']

Ra_values = new_comaMb['RA_degrees']
Dec_values = new_comaMb['Dec_degress']

ra = ngc['ra']
dec = ngc['de']

# Create a scatter plot
plt.scatter(x_values, y_values, color='b', marker = '*', label = 'Data2')
plt.scatter(ra_values, dec_values, color='b', marker = '*', label = 'Data1')
plt.scatter(Ra_values, Dec_values, color='b', marker = '*', label = 'Data3')
plt.scatter(ra, dec, color='r', marker = '*')

plt.annotate('NGC4889', xy=(195.1, 29))
plt.annotate('NGC 4874', xy=(194.87,28.9))
plt.annotate('NGC 4839', xy=(194.4,28.2))
# Set the title of the plot
plt.xlim(195.4,194)
plt.ylim(27.6,29.5)

degree_sign = u'\N{DEGREE SIGN}'
plt.xlabel('RA [' + degree_sign + ']')
plt.ylabel('Dec [' + degree_sign + ']')

# Display the plot
plt.show()


# In[63]:


ra = ngc['ra']
dec = ngc['de']

plt.scatter(ra, dec, color='r', marker = '*')
plt.annotate('NGC4889', xy=(195.1, 29))
plt.annotate('NGC 4874', xy=(194.87,28.9))
plt.annotate('NGC 4839', xy=(194.4,28.2))
# Set the title of the plot
plt.xlim(195.4,194)
plt.ylim(27.6,29.5)

degree_sign = u'\N{DEGREE SIGN}'
plt.xlabel('RA [' + degree_sign + ']')
plt.ylabel('Dec [' + degree_sign + ']')

plt.show()


# In[64]:


import matplotlib.pyplot as plt
import matplotlib.patches as patches

# create a figure and a subplot
fig, ax = plt.subplots()

# plot the data points
ax.scatter(ra, dec, color='r', marker='*')

# add a rectangle patch for the first rectangle
rect1_patch = patches.Rectangle((194.6, 28.5), 0.6, 1, fill=False, edgecolor='blue', label ='Coma1')
ax.add_patch(rect1_patch)

# add a rectangle patch for the second rectangle
rect2_patch = patches.Rectangle((194.0, 27.55), 0.6, 1, fill=False, edgecolor='r', label = 'Coma3')
ax.add_patch(rect2_patch)

plt.annotate('NGC4889', xy=(195.1, 29))
plt.annotate('NGC 4874', xy=(194.87,28.9))
plt.annotate('NGC 4839', xy=(194.4,28.2))

# set the limits of the subplot
ax.set_xlim(195.3,193.9)
ax.set_ylim(27.6,29.5)

# set the labels of the subplot
degree_sign = u'\N{DEGREE SIGN}'
ax.set_xlabel('RA [' + degree_sign + ']')
ax.set_ylabel('Dec [' + degree_sign + ']')

# show the plot
plt.legend()
plt.show()

