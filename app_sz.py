import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

import healpy as hp

import scipy as sp

import altair as alt

from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord

import ipywidgets as widgets

from scipy.integrate import quad

import seaborn as sns

import streamlit as st

# Third-party dependencies
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from IPython.display import Image
import astropy.coordinates as coord

sns.set_style("darkgrid")

# ==============================================

from PIL import Image
import requests
from io import BytesIO

response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/cmb-1.jpg")
img = Image.open(BytesIO(response.content))

# ===============================================

#image = Image.open('/home/nareg/Downloads/cmb-1.jpg')
st.image(img, use_column_width = True)


catalog = os.path.join(".", "Catalogs", "HFI_PCCS_SZ-union_R2.08.fits")

hdulist_union = fits.open(catalog)

st.markdown('## **Planck _Sunyaev-Zel\'dovich_ Galaxy Cluster Catalog**')

df0 = pd.DataFrame({
                   'snr': range(0, 40, 1)
                   })

df1 = pd.DataFrame({
                   'zmin': np.arange(0.0, 1.0, 0.1)
                   })

df2 = pd.DataFrame({
                   'show_data': [True, False],
                   'show_map': [True, False]
                   })


#
#snr_min = st.sidebar.selectbox(
#    'Minimum signal to noise of detection: ',
#    df0['snr'])

snr_min = st.slider('Minimum Signal to Noise Ratio (s/n): ', 3, 15, 5)
z_min = st.slider('Minimum Redshift: ', 0.0, 1.0, 0.5)

# Changing RA and dec to Lon, lat
def rad_2_gtic(ra, dec):

    """
    Convert RA and dec to galactiv Lon and Lat

    """
    g = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='fk5')

    return g.galactic.l.value, g.galactic.b.value


def gtic_2_rad(lon, lat):

    """
    Convert Lon and Lat to RA and dec

    """
    g = SkyCoord(lon*u.degree, lat*u.degree, frame='galactic')

    return g.fk5.ra.value, g.fk5.dec.value


cosmo_sample = st.sidebar.selectbox(
    'Cosmology sample: ',
    df2['show_data'])

# Clusters used in Cosmology
if cosmo_sample:

    cosmo_id = np.where(hdulist_union[1].data['COSMO'] == True)[0]
    z_ = np.asanyarray(hdulist_union[1].data['REDSHIFT'][cosmo_id], dtype = np.float)
    z = z_[z_ >= 0.0]
    glon = np.asanyarray(hdulist_union[1].data['GLON'][cosmo_id], dtype = np.float)
    glon = glon[z_ >= 0.0]
    glat = np.asanyarray(hdulist_union[1].data['GLAT'][cosmo_id], dtype = np.float)
    glat = glat[z_ >= 0.0]
    q = np.asanyarray(hdulist_union[1].data['SNR'][cosmo_id], dtype = np.float)
    q = q[z_ >= 0.0]

    Y5R500 = np.asanyarray(hdulist_union[1].data['Y5R500'][cosmo_id], dtype = np.float)
    Y5R500 = Y5R500[z_ >= 0.0]

    Msz = np.asanyarray(hdulist_union[1].data['MSZ'][cosmo_id], dtype = np.float)
    Msz = Msz[z_ >= 0.0]

else:

    cosmo_id = np.where(hdulist_union[1].data['REDSHIFT'] >= 0.00)[0]
    glon = np.asanyarray(hdulist_union[1].data['GLON'][cosmo_id], dtype = np.float)
    glat = np.asanyarray(hdulist_union[1].data['GLAT'][cosmo_id], dtype = np.float)
    q = np.asanyarray(hdulist_union[1].data['SNR'][cosmo_id], dtype = np.float)
    z = np.asanyarray(hdulist_union[1].data['REDSHIFT'][cosmo_id], dtype = np.float)
    Y5R500 = np.asanyarray(hdulist_union[1].data['Y5R500'][cosmo_id], dtype = np.float)/1e3
    Msz = np.asanyarray(hdulist_union[1].data['MSZ'][cosmo_id], dtype = np.float)


data = {'z': z, 'SNR': q, 'GLON': glon, 'GLAT': glat, 'M_sz/1e14': Msz, 'Y5R500': Y5R500}

df = pd.DataFrame(data)

df = df[(df.z.values >= z_min) & (df.SNR.values >= snr_min)]

len(df), "Galaxy Clusters with signal to noise ratio (s/n) larger than ", snr_min, '.'
"Largest mass with s/n > ", snr_min, ' : ', str(np.round(Msz.max(), 2)), r'$\times \ 10^{14}$'



z_bin = np.arange(0, 1, .1)
q_bin = np.arange(0.0, 0.25*5, 0.25) + np.log10(6.007906436920166)

df2 = pd.DataFrame({
                   'show_data': [True, False],
                   'show_map': [True, False]
                   })

show_table = st.sidebar.selectbox(
    'Show table: ',
    df2['show_data'])

show_map = st.sidebar.selectbox(
    'Show map: ',
    df2['show_map'])


# ======================================================================================================

if show_map:

    deg = df['GLON']#glon
    deg1 = df['GLAT']

    ra = coord.Angle(deg*u.degree)
    ra = ra.wrap_at(180*u.degree)
    dec = coord.Angle(deg1*u.degree)

    #fig = plt.figure()
    #hp.mollview(np.zeros(hp.nside2npix(6)), min = z.min(), max = z.max())
    #hp.graticule()
    #ax = fig.add_subplot(111, projection = "mollweide")
    #hp.projscatter(glon[np.where(q > snr_min)[0]], glat[np.where(q > snr_min)[0]], lonlat = True, c = z[np.where(q > snr_min)[0]], s = 20, edgecolor = 'k')            # Add grid
    #hp.mollview(np.zeros(hp.nside2npix(6)), cbar = None)
    fig = plt.figure()
    #fig = plt.gcf()
    ax = plt.gca(projection = "mollweide", facecolor = "LightCyan")
    #ax.scatter(ra.radian[np.where(q > snr_min)[0]], dec.radian[np.where(q > snr_min)[0]], marker = '*', c = -1*z[np.where(q > snr_min)[0]], s = 20)
    image = ax.scatter(ra.radian, dec.radian, marker = 'o', c = df.z, s = 7, cmap = 'jet')#ax.get_images()[0]
    cmap = fig.colorbar(image, ax = ax, shrink = 0.4)
    cmap.ax.set_ylabel('Redshift', rotation = 270, labelpad = 14)
    plt.grid(True, color = 'black', alpha = 0.2)
    #fig.colorbar(z)
    plt.tight_layout()
    st.pyplot()

st.markdown("""
                Made by [Nareg Mirzatuny](https://github.com/NaregM)                                     
Source code: [GitHub](
                https://github.com/NaregM/coin_simulator) 
                
""")

# ======================================================================================================
c = alt.Chart(df).mark_circle().encode(alt.X('z', scale=alt.Scale(zero=False)),
                                        alt.Y('M_sz/1e14', scale=alt.Scale(zero=False)),
                                        size = 'Y5R500', color=alt.Color('SNR', scale=alt.Scale(scheme='darkblue')), tooltip = ['z', 'M_sz/1e14', 'SNR'])

st.altair_chart(c, use_container_width = True)

print()
# ======================================================================================================
fig, ax = plt.subplots(dpi = 30)

ax.hist(df.z.values,bins = 12, histtype = "step", lw = 2, color = "deepskyblue")

ax.set_xlabel('z', size = 12)
ax.set_ylabel(r'$\mathrm{N(z)}$', size = 12)
ax.set_title('Redshift Distribution of Clusters With $s/n$ > %.2f' % snr_min, size = 12)
ax.set_xlim(z_min+0.02, 1)

ax.margins(0.3)
plt.tight_layout()
st.pyplot(fig)

# ======================================================================================================
if show_table:

    df
