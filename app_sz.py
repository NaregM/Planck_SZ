import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

import healpy as hp

import altair as alt

from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord

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


#image = Image.open('/home/nareg/Downloads/cmb-1.jpg')
st.image(img, use_column_width = True)
# ===============================================

df_cat = pd.DataFrame({
                     'union': ["HFI_PCCS_SZ-union_R2.08.fits"]
                     })

catalog = st.sidebar.selectbox(
            'Catalogs: ',
            df_cat['union'])


path = os.path.join(".", "Catalogs", catalog)

hdulist_union = fits.open(path)

st.markdown('## **Planck _Sunyaev-Zel\'dovich_ Galaxy Cluster Catalog Explorer**')

st.markdown('This app provides interactive visual reports and statistical summaries of the SZ galaxy clusters dicovered by the Planck satallite.')
st.markdown('http://pla.esac.esa.int')
st.markdown('-------------------------------------------------------------------------------')


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

snr_min = st.slider('Minimum Signal to Noise Ratio (s/n) of Detection: ', 3, 15, 5)
z_min = st.slider('Minimum Redshift: ', 0.0, 1.0, 0.15)

# Clusters used in Cosmology
if st.checkbox('Use galaxy clusters from the cosmology sample'):

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


z_bin = np.arange(0, 1, .1)

df2 = pd.DataFrame({
                   'show_data': [True, False],
                   'show_map': [True, False]
                   })

show_table = st.sidebar.selectbox(
    'Show Table: ',
    df2['show_data'])

show_map = st.sidebar.selectbox(
    'Show Sky Map: ',
    df2['show_map'])

show_2d = st.sidebar.selectbox(
    'Show 2D Distribution (z, s/n): ',
    df2['show_data'])

'*', len(df), "Galaxy Clusters with signal to noise ratio (s/n) larger than ", snr_min, '.'

dfe = df[['z', 'SNR', 'M_sz/1e14', 'Y5R500']]
df.style.set_caption("Hello World")
st.table(dfe.describe()[1:])
# ======================================================================================================
if show_map:

    deg = df['GLON']#glon
    deg1 = df['GLAT']

    ra = coord.Angle(deg*u.degree)
    ra = ra.wrap_at(180*u.degree)
    dec = coord.Angle(deg1*u.degree)

    fig = plt.figure(dpi = 1500)

    ax = plt.gca(projection = "mollweide", facecolor = "deepskyblue")
    image = ax.scatter(ra.radian, dec.radian, marker = 'o', c = df.z, s = 7, cmap = 'jet')
    cmap = fig.colorbar(image, ax = ax, shrink = 0.4)
    cmap.ax.set_ylabel('Redshift', rotation = 270, labelpad = 12)
    plt.grid(True, color = 'black', alpha = 0.2)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown('---------------------------------------------------------------------------')

# ======================================================================================================
st.markdown('Interactive plot that summaries four main characteristic of each cluster: redshit, s/n of detection, $M_{sz}$ and $Y_{5R500}$')


c = alt.Chart(df).mark_circle().encode(alt.X('z', scale=alt.Scale(zero=False)),
                                            alt.Y('M_sz/1e14', scale=alt.Scale(zero=False)),
                                             color=alt.Color('SNR', scale=alt.Scale(scheme='darkblue')), tooltip = ['z', 'M_sz/1e14', 'SNR'])

if st.checkbox('Add Linear Regression'):

    c = alt.Chart(df).mark_circle().encode(alt.X('z', scale=alt.Scale(zero=False)),
                                                alt.Y('M_sz/1e14', scale=alt.Scale(zero=False)),
                                                size = 'Y5R500', color=alt.Color('SNR', scale=alt.Scale(scheme='darkblue')), tooltip = ['z', 'M_sz/1e14', 'SNR', 'Y5R500'])

    c1 = alt.Chart(df).mark_circle().encode(alt.X('z'), alt.Y('M_sz/1e14'))

    reg_plot = c1.transform_regression('z', 'M_sz/1e14').mark_line(color = 'red')
    st.altair_chart(c + reg_plot, use_container_width = True)

else:

    c = alt.Chart(df).mark_circle().encode(alt.X('z', scale=alt.Scale(zero=False)),
                                                alt.Y('M_sz/1e14', scale=alt.Scale(zero=False)),
                                                size = 'Y5R500', color=alt.Color('SNR', scale=alt.Scale(scheme='darkblue')), tooltip = ['z', 'M_sz/1e14', 'SNR', 'Y5R500'])
    st.altair_chart(c, use_container_width = True)

# ======================================================================================================
st.markdown('---------------------------------------------------------------------------')

fig, ax = plt.subplots(dpi = 13)

ax.hist(df.z.values,bins = 12, histtype = "step", lw = 2, color = "deepskyblue")

ax.set_xlabel('z', size = 12)
ax.set_ylabel(r'$\mathrm{N(z)}$', size = 12)
ax.set_title('Redshift Distribution of Clusters With $s/n$ > %.2f' % snr_min, size = 12)
ax.set_xlim(z_min+0.02, 1)

ax.margins(0.3)
plt.tight_layout()
st.pyplot(fig)


if show_2d:

    st.markdown('---------------------------------------------------------------------------')
    st.markdown('A representation of 2-dimensional distribution (z VS s/n). Majority of clusters are at lower redshifts and have smaller signal to noise ratio.')

    z_bin = np.linspace(z_min, 1, 12)
    snr_bin = np.linspace(snr_min, 40, 12)

    Nij_snr = np.zeros((12, 12))

    for i in range(1, z_bin.size):

        for j in range(1, snr_bin.size):

            Nij_snr[i-1, j-1] = np.sum((z_bin[i-1] <= df.z.values) & (df.z.values < z_bin[i]) &
                            (snr_bin[j-1] <= df.SNR.values) & ((df.SNR.values) < snr_bin[j]))


    plt.figure()
    plt.imshow(Nij_snr, origin = 'lower', cmap = plt.cm.jet, extent = [4, 45, 0.0, 1.0], aspect = 'auto')
    plt.xlabel('SNR')
    plt.ylabel('z')
    plt.colorbar(label = 'Number of galaxy clusters')
    plt.grid(False)
    st.pyplot()

# ======================================================================================================
if show_table:
    st.markdown('---------------------------------------------------------------------------')
    df




st.markdown('-------------------------------------------------------------------------------')

st.markdown("""
                Made by [Nareg Mirzatuny](https://github.com/NaregM)

Source code: [GitHub](
                https://github.com/NaregM/planck_sz)

""")
st.markdown('-------------------------------------------------------------------------------')

