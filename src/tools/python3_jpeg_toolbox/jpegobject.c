/* $Id: jpegobject.c 1479 2009-04-29 06:31:13Z css1hs $
 * (C) 2010: University of Surrey, UK
 * Author: Hans Georg Schaathun <georg@schaathun.net>
 * Inspired by Phil Sallee's toolbox for Matlab 6/2003.
 *
 * Depends on the jpeglib of the Independent JPEG Group, see separate
 * copyright notice for this library.
 */

#include <stdio.h>
#include <stdlib.h>
#include "Python.h"
#include "structmember.h"
#include "numpy/arrayobject.h"
#include <setjmp.h>

/* The following have been modified to avoid conflicts */
#include "jpeglib/jerror.h"
#include "jpeglib/jpeglib.h"
//#include "jpeglib/jpegint.h"
//#include "jerror.h"
//#include "jpeglib.h"
//#include "jpegint.h"

/* We need to create our own error handler so that we can override the 
 * default handler in case a fatal error occurs.  The standard error_exit
 * method calls exit() which doesn't clean things up properly and also 
 * exits Matlab.  This is described in the example.c routine provided in
 * the IJG's code library.
 */
struct my_error_mgr {
  struct jpeg_error_mgr pub;	/* "public" fields */
  jmp_buf setjmp_buffer;	/* for return to caller */
};

typedef struct my_error_mgr * my_error_ptr;

METHODDEF(void)
my_error_exit (j_common_ptr cinfo)
{
  char buffer[JMSG_LENGTH_MAX];

  /* cinfo->err really points to a my_error_mgr struct, so coerce pointer */
  my_error_ptr myerr = (my_error_ptr) cinfo->err;

  /* create the message */
  (*cinfo->err->format_message) (cinfo, buffer);
  printf("Error: %s\n",buffer);
  
  /* return control to the setjmp point */
  longjmp(myerr->setjmp_buffer, 1);
} ;

typedef struct {
  PyObject_HEAD
  int image_width ;          /* image width in pixels */
  int image_height ;         /* image height in pixels */
  int image_components ;     /* number of image color components */
  int image_color_space ;    /* in/out_color_space */
  int jpeg_components ;      /* number of JPEG color components */
  int jpeg_color_space ;     /* color space of DCT coefficients */
  /* int image_num_colors ;     *//* color depth of input/output image */
  PyObject * comments ;             /* COM markers, if any */
  PyObject * coef_arrays ;          /* DCT arrays for each component */
  PyObject * quant_tables ;         /* quantization tables */
  int optimize_coding ;      /* flag to optimize huffman tables */
  PyObject * comp_info ;            /* component info struct array */
  int progressive_mode ;    /* is progressive mode */
  PyObject *filename ;
} jpegObject ;

typedef struct {
  PyObject_HEAD
  int component_id ;         /* JPEG one byte identifier code */
  int h_samp_factor ;        /* horizontal sampling factor */
  int v_samp_factor ;        /* vertical sampling factor */
  int quant_tbl_no ;         /* quantization table number for component */
  int dc_tbl_no ;            /* DC entropy coding table number */
  int ac_tbl_no ;            /* AC entropy encoding table number */
} compInfoObject ;

/* Destructor */
static void jpegDel( jpegObject *self )
{
  Py_XDECREF ( self->comp_info ) ;
  Py_XDECREF ( self->quant_tables ) ;
  Py_XDECREF ( self->comments ) ;
  Py_XDECREF ( self->coef_arrays ) ;
  Py_XDECREF ( self->filename ) ;
  Py_TYPE(self)->tp_free( (PyObject*) self ) ;
}

static PyObject *jpegSave( jpegObject *self, PyObject *args, PyObject *kw )
{
  static char *kwlist[] = {"file", NULL};
  char *filename = NULL ;
  FILE *file ;
  int n, qn ; /* counter */
  jvirt_barray_ptr *coef_arrays = NULL;
  jpeg_component_info *compptr;

  struct jpeg_compress_struct cinfo;
  struct my_error_mgr jerr;

  if (!PyArg_ParseTupleAndKeywords(args, kw, "|s", kwlist, &filename ))
     return NULL ; 
  if ( filename == NULL ) filename = PyBytes_AsString ( self->filename ) ;
  else {
    Py_XDECREF ( self->filename ) ;  /* Discard old string */
    self->filename = Py_BuildValue ( "s", filename ) ;
    /* Py_BuildValue() eturns new reference */
  }

  if ((file = fopen(filename, "wb")) == NULL) {
    PyErr_SetString(PyExc_IOError, "Can't open file") ;
    return NULL ;
  }

  //printf("DCT_scaled_size: %d\n",cinfo.comp_info->DCT_scaled_size);
  
  /* set up the normal JPEG error routines, then override error_exit. */
  cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;

  /* establish the setjmp return context for my_error_exit to use. */
  if (setjmp(jerr.setjmp_buffer)) {
    jpeg_destroy_compress(&cinfo);
    fclose(file);
    PyErr_SetString(PyExc_IOError, "Error writing file");
    return NULL ;
    }

  /* initialize JPEG decompression object */
  jpeg_create_compress(&cinfo);

  /* write the output file */
  jpeg_stdio_dest(&cinfo, file);

/* Set scalars */
  cinfo.image_width = self->image_width ;
  cinfo.image_height = self->image_height ;
  cinfo.input_components = self->image_components ;
  cinfo.in_color_space = self->image_color_space ; 

  jpeg_set_defaults(&cinfo);

  cinfo.optimize_coding = self->optimize_coding ;
  cinfo.num_components = self->jpeg_components ;
  cinfo.jpeg_color_space = self->jpeg_color_space ;
  cinfo.progressive_mode = self->progressive_mode ;

/* Set Component Info */
  for (n = 0; n < cinfo.num_components; n++)
  {
    PyObject *D ;
    long L ;
    D = PyList_GetItem ( self->comp_info, n ) ;
    cinfo.comp_info[n].component_id = 
       L = PyLong_AsLong ( PyDict_GetItemString ( D, "component_id" ) ) ;
    if ( L == -1 && PyErr_Occurred() ) return NULL ;
    cinfo.comp_info[n].h_samp_factor =
       L = PyLong_AsLong (PyDict_GetItemString ( D, "h_samp_factor" ) ) ;
    if ( L == -1 && PyErr_Occurred() ) return NULL ;
    cinfo.comp_info[n].v_samp_factor =
       L = PyLong_AsLong (PyDict_GetItemString ( D, "v_samp_factor" ) ) ;
    if ( L == -1 && PyErr_Occurred() ) return NULL ;
    cinfo.comp_info[n].quant_tbl_no =
       L = PyLong_AsLong (PyDict_GetItemString ( D, "quant_tbl_no" ) ) ;
    if ( L == -1 && PyErr_Occurred() ) return NULL ;
    cinfo.comp_info[n].ac_tbl_no =
       L = PyLong_AsLong (PyDict_GetItemString ( D, "ac_tbl_no" ) ) ;
    if ( L == -1 && PyErr_Occurred() ) return NULL ;
    cinfo.comp_info[n].dc_tbl_no =
       L = PyLong_AsLong (PyDict_GetItemString ( D, "dc_tbl_no" ) ) ;
    if ( L == -1 && PyErr_Occurred() ) return NULL ;

  }

/* Set (empty) Huffmann Tables */
  for ( n=0 ; n < NUM_HUFF_TBLS; n++) cinfo.ac_huff_tbl_ptrs[n] = NULL;
  for ( n=0 ; n < NUM_HUFF_TBLS; n++) cinfo.dc_huff_tbl_ptrs[n] = NULL;

/* Coefficient Arrays */
  /* request virtual block arrays */
  coef_arrays = (jvirt_barray_ptr *)
    (cinfo.mem->alloc_small) ((j_common_ptr) &cinfo, JPOOL_IMAGE,
     sizeof(jvirt_barray_ptr) * cinfo.num_components);
  for (n = 0; n < cinfo.num_components; n++) {
    int c_height, c_width ; 
    PyObject *C0  ;
    PyArrayObject *C ;

    C0 = PyList_GetItem ( self->coef_arrays, n ) ;
    C  = (PyArrayObject *) C0 ;
    compptr = cinfo.comp_info + n;

    c_height = C->dimensions[0] ;
    c_width = C->dimensions[1] ;
    compptr->height_in_blocks = c_height / DCTSIZE;
    compptr->width_in_blocks = c_width / DCTSIZE;

    coef_arrays[n] = (cinfo.mem->request_virt_barray)
      ((j_common_ptr) &cinfo, JPOOL_IMAGE, TRUE,
       (JDIMENSION) jround_up((long) compptr->width_in_blocks,
                              (long) compptr->h_samp_factor),
       (JDIMENSION) jround_up((long) compptr->height_in_blocks,
                              (long) compptr->v_samp_factor),
       (JDIMENSION) compptr->v_samp_factor);
  }

  /* realize virtual block arrays */
  jpeg_write_coefficients(&cinfo,coef_arrays);

  /* populate the array with the DCT coefficients */
  for (n = 0; n < cinfo.num_components; n++) {
    int c_height, c_width, i, j ; 
    JBLOCKARRAY buffer;
    JDIMENSION blk_x, blk_y ; 
    JCOEFPTR bufptr; 
    int *mp ;
    PyObject *C0 ;
    PyArrayObject *C ;
    compptr = cinfo.comp_info + n;

    /* Get a pointer to the mx coefficient array */

    C0 = PyList_GetItem ( self->coef_arrays, n ) ;
    C  = (PyArrayObject *) C0 ;
    mp = (int *) C->data ;
    
    c_height = C->dimensions[0] ;
    c_width = C->dimensions[1] ;

    /* Copy coefficients to virtual block arrays */
    for (blk_y = 0; blk_y < compptr->height_in_blocks; blk_y++) {
      buffer = (cinfo.mem->access_virt_barray)
  	((j_common_ptr) &cinfo, coef_arrays[n], blk_y, 1, TRUE);

      for (blk_x = 0; blk_x < compptr->width_in_blocks; blk_x++) {
        bufptr = buffer[0][blk_x];
        for (i = 0; i < DCTSIZE; i++) {       /* for each row in block */
          for (j = 0; j < DCTSIZE; j++) {     /* for each column in block */
	    int x, y ;
	    y = DCTSIZE*blk_y + i ;
	    x = DCTSIZE*blk_x + j ;
            /* bufptr[i*DCTSIZE+j] = (JCOEF) mp[j*c_height+i];  */
	    /* T = C->descr->getitem( mp + y*n1 + x*n2 ) ;
	       bufptr[i*DCTSIZE+j] = (JCOEF) PyLong_AsLong ( T ) ;  */
            bufptr[i*DCTSIZE+j] =  (JCOEF) mp[y*c_width + x] ;  
	    /* Corresponding Init line:
	     *     mp[y*dim[1] + x] = (int) bufptr[i*DCTSIZE+j] ; */
	  }
	}
        /* mp+=DCTSIZE*c_height; */
      }
      /* mp=(mptop+=DCTSIZE); */
    }
  }

  /* get the quantization tables */
  /* qn = PyList_Size ( self->quant_tables ) ; */
  for (n = 0 ; n < NUM_QUANT_TBLS ; n++) {
    PyObject *C0 ;
    PyArrayObject *C ;
    int *mp, i, j ;

    /* Allocate space if necessary */
    if (cinfo.quant_tbl_ptrs[n] == NULL)
      cinfo.quant_tbl_ptrs[n] = jpeg_alloc_quant_table((j_common_ptr) &cinfo);

    /* Fill the table */
    C0 = PyList_GetItem ( self->quant_tables, n ) ;

    if ( C0 == Py_None ) {
      cinfo.quant_tbl_ptrs[n] = NULL;
      continue ;
    }

    C  = (PyArrayObject *) C0 ;
    mp = (int *) C->data ;
    compptr = cinfo.comp_info + n;
    
    for (i = 0; i < DCTSIZE; i++) 
      for (j = 0; j < DCTSIZE; j++) {
        int t ;

        t = mp[i*DCTSIZE+j];

        if (t<1 || t>65535) {
          PyErr_SetString(PyExc_IOError, 
             "Quantisation table entries out of range 1..65535"  );
	  return NULL ;
        }

        cinfo.quant_tbl_ptrs[n]->quantval[i*DCTSIZE+j] = (UINT16) t;
      }
  }

  /* copy markers */
  qn = PyList_Size(self->comments);
  for (n = 0; n < qn; n++) {
    char *comment ;
    PyObject *D ;
    int slen ;
    D = PyList_GetItem ( self->comments, n ) ;
    slen = PyBytes_Size(D) ;
    comment = PyBytes_AsString(D) ;
    printf ( "jpegSave() Comment(%i)\n", n ) ;
    printf ( "%s\n", comment ) ;
    jpeg_write_marker(&cinfo, JPEG_COM, comment, slen );
    /* Do we have to copy the string ?? */
  }

  /* Clean up */
  jpeg_finish_compress(&cinfo); 
  jpeg_destroy_compress(&cinfo);
  fclose( file ) ;

  Py_RETURN_NONE ;
} /* jpegSave */

/* Constructor */
static int jpegInit( jpegObject *self, PyObject *args, PyObject *kw )
{
  struct jpeg_decompress_struct cinfo;
  struct my_error_mgr jerr;
  jvirt_barray_ptr *coef_arrays;
  jpeg_saved_marker_ptr marker_ptr;
  FILE *infile;
  /* int strlen, ci, i, j, n, dims[2]; */
  int n, i, j ; /* loop indices */
  char *filename;
  /* mxArray *mxtemp, *mxjpeg_obj, *mxcoef_arrays, *mxcomments; */
  /* mxArray *mxquant_tables, *mxhuff_tables, *mxcomp_info; */
  static int quant_dim[] = { DCTSIZE, DCTSIZE, 0 } ;

  static char *kwlist[] = {"file", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kw, "s", kwlist, &filename ))
     return -1; 

  self->filename = Py_BuildValue ( "s", filename ) ;
  /* Py_BuildValue() eturns new reference */

  /* open file */
  if ((infile = fopen(filename, "rb")) == NULL) {
    PyErr_SetString(PyExc_IOError, "Can't open file") ;
    return -1 ;
  }

  /* set up the normal JPEG error routines, then override error_exit. */
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = my_error_exit;

  /* establish the setjmp return context for my_error_exit to use. */
  if (setjmp(jerr.setjmp_buffer)) {
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    PyErr_SetString(PyExc_IOError, "Error reading file");
    return -1 ;
  }

  /* initialize JPEG decompression object */
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, infile);

  /* save contents of markers */
  jpeg_save_markers(&cinfo, JPEG_COM, 0xFFFF);

  /* read header and coefficients */
  jpeg_read_header(&cinfo, TRUE);

  /* for some reason out_color_components isn't being set by
     jpeg_read_header, so we will infer it from out_color_space: */
  switch (cinfo.out_color_space) {
    case JCS_GRAYSCALE: cinfo.out_color_components = 1; break;
    case JCS_RGB:       cinfo.out_color_components = 3; break;
    case JCS_YCbCr:     cinfo.out_color_components = 3; break;
    case JCS_CMYK:      cinfo.out_color_components = 4; break;
    case JCS_YCCK:      cinfo.out_color_components = 4; break;
    default:
      jpeg_destroy_decompress(&cinfo);
      fclose(infile);
      PyErr_SetString(PyExc_IOError, "Unknown Colour Space");
      return -1 ;
    break ;
  }

  /* copy header information */
  self->image_width       = cinfo.image_width ;
  self->image_height      = cinfo.image_height ;
  self->image_color_space = cinfo.out_color_space ;
  self->image_components  = cinfo.out_color_components ;
  self->jpeg_color_space  = cinfo.jpeg_color_space ;
  self->jpeg_components   = cinfo.num_components ;
  self->progressive_mode  = cinfo.progressive_mode ;


  /* set optimize_coding flag for jpeg_write() */
  /* mxSetField(mxjpeg_obj,0,"optimize_coding",mxCDS(FALSE)); */
  self->optimize_coding = TRUE ;
  /* This will force generation of new Huffmann tables */

#define PDsetInt(D,k,i)  \
    PyDict_SetItemString( D, k, T = Py_BuildValue( "i", i ) ) ;\
    Py_XDECREF ( T )
  Py_XDECREF ( self->comp_info ) ;

  /* copy component information */
  self->comp_info = PyList_New(cinfo.num_components) ;
  for (n = 0; n < cinfo.num_components; n++) {
    PyObject *D, *T ;
    D = PyDict_New() ;
    PyList_SetItem(self->comp_info, n, D) ;
    PDsetInt( D, "component_id", cinfo.comp_info[n].component_id );
    PDsetInt( D, "h_samp_factor", cinfo.comp_info[n].h_samp_factor ) ; 
    PDsetInt( D, "v_samp_factor", cinfo.comp_info[n].v_samp_factor ) ; 
    PDsetInt( D, "quant_tbl_no", cinfo.comp_info[n].quant_tbl_no ) ; 
    PDsetInt( D, "ac_tbl_no", cinfo.comp_info[n].ac_tbl_no ) ; 
    PDsetInt( D, "dc_tbl_no", cinfo.comp_info[n].dc_tbl_no ) ; 
  }

  /* copy markers */
  self->comments = PyList_New(0) ;
  marker_ptr = cinfo.marker_list;
  while (marker_ptr != NULL) {
    PyObject *C ;
    switch (marker_ptr->marker) {
      case JPEG_COM:
         /* for (i = 0; i < (int) marker_ptr->data_length; i++) 
           *mcp++ = (mxChar) marker_ptr->data[i]; */
	 C = Py_BuildValue( "s#", marker_ptr->data, marker_ptr->data_length );
	 PyList_Append( self->comments, C ) ;
	 Py_DECREF(C) ; 
         break;
      default:
         break;
    }
    marker_ptr = marker_ptr->next;
  }

  /* copy the quantization tables */
  self->quant_tables = PyList_New(NUM_QUANT_TBLS);
  for (n = 0; n < NUM_QUANT_TBLS; n++) {
    if (cinfo.quant_tbl_ptrs[n] != NULL) {
      PyObject *Q ;
      PyArrayObject *Q0 ;
      JQUANT_TBL *quant_ptr;
      int *mp ;
      quant_ptr = cinfo.quant_tbl_ptrs[n];
      Q =  PyArray_FromDims ( 2, quant_dim, PyArray_INT ) ;
      Q0 = (PyArrayObject *) Q ;
      mp = (int*) Q0->data ;
      for (i = 0; i < DCTSIZE; i++) 
        for (j = 0; j < DCTSIZE; j++) {
	  mp[i*DCTSIZE + j] = (int) quant_ptr->quantval[i*DCTSIZE+j] ;
        }
      PyList_SetItem( self->quant_tables, n, (PyObject*) Q ) ;
    } else {
      Py_INCREF( Py_None ) ;
      PyList_SetItem( self->quant_tables, n, Py_None ) ;
    }
  }

  /* creation and population of the DCT coefficient arrays */
  coef_arrays = jpeg_read_coefficients(&cinfo) ;
  self->coef_arrays = PyList_New( cinfo.num_components );
  for (n = 0; n < cinfo.num_components; n++) {
    PyObject *Q ;
    PyArrayObject *Q0 ;
    jpeg_component_info *compptr;
    JBLOCKARRAY buffer;
    JDIMENSION blk_x, blk_y ; 
    int dim[2] ;
    int *mp ;
    JCOEFPTR bufptr; 


    compptr = cinfo.comp_info + n;
    dim[0] = compptr->height_in_blocks * DCTSIZE;
    dim[1] = compptr->width_in_blocks * DCTSIZE;

    Q =  PyArray_FromDims ( 2, dim, PyArray_INT ) ;
    Q0 = (PyArrayObject *) Q ;
    mp = (int *) Q0->data ;

    /* printf ( "coef_array n=%i (%ix%i)\n", n, dim[0], dim[1] ) ; */

    for (blk_y = 0 ; blk_y < compptr->height_in_blocks ; blk_y++) {
      buffer = (cinfo.mem->access_virt_barray)
    	((j_common_ptr) &cinfo, coef_arrays[n], blk_y, 1, FALSE);
      for (blk_x = 0; blk_x < compptr->width_in_blocks; blk_x++) {
        bufptr = buffer[0][blk_x] ;
        for (i = 0; i < DCTSIZE; i++) {  /* for each row in block */
          for (j = 0; j < DCTSIZE; j++) { /* for each column in block */
	  /* printf ( "(%ix%i) ", dim[0], dim[1] ) ;
             printf ( "[%i,%i;%i,%i]\n", blk_y, blk_x, i, j ) ; */
	    unsigned int x, y ; 
	    y = DCTSIZE*blk_y + i ;
	    x = DCTSIZE*blk_x + j ;
	    mp[y*dim[1] + x] = (int) bufptr[i*DCTSIZE+j] ;
	  }
	}
      }
    }
    PyList_SetItem( self->coef_arrays, n, Q ) ;
  }

  /* done with cinfo */
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);

  /* close input file */
  fclose(infile);

  return 0 ;

}

static PyObject * printColourCodes(
    jpegObject *self, PyObject *args, PyObject *kw 
) {
  printf ( "JCS_GRAYSCALE\t = %i\n", JCS_GRAYSCALE ) ;
  printf ( "JCS_RGB\t = %i\n", JCS_RGB ) ;
  printf ( "JCS_YCbCr\t = %i\n", JCS_YCbCr ) ;
  printf ( "JCS_CMYK\t = %i\n", JCS_CMYK ) ;
  printf ( "JCS_YCCK\t = %i\n", JCS_YCCK ) ;
  Py_RETURN_NONE ;
}

static PyMethodDef
jpegObject_methods[] = {
  { "save", (PyCFunction) jpegSave, METH_VARARGS|METH_KEYWORDS, 
    "Save the image back to file. (filename optional)" },
  { "printColourCodes",
    (PyCFunction) printColourCodes, METH_VARARGS|METH_KEYWORDS, 
    "Print the colour space codes used." },
  { NULL, NULL, 0, NULL }
} ;

static PyMemberDef
jpegObject_members[] = {
  { "image_width", T_INT, offsetof(jpegObject,image_width), 0,
    "The width of the image" },
  { "image_height", T_INT, offsetof(jpegObject,image_height), 0,
    "The height of the image" },
  { "image_components", T_INT, offsetof(jpegObject,image_components), 0,
    "Number of colour components" },
  { "image_color_space", T_INT, offsetof(jpegObject,image_color_space), 0,
    "Color space of the decompressed image" },
  { "jpeg_components", T_INT, offsetof(jpegObject,jpeg_components), 0,
    "Number of colour components in the JPEG representation" },
  { "jpeg_color_space", T_INT, offsetof(jpegObject,jpeg_color_space), 0,
    "Color space of the JPEG representation" },
  { "optimize_coding", T_INT, offsetof(jpegObject,optimize_coding), 0,
    "Flag to say if the Huffmann tables should be optimized on saving. "
    "This should always be 1 (TRUE)." },
  { "progressive_mode", T_INT, offsetof(jpegObject,progressive_mode), 0,
    "Flag to say if Progressive mode should be used." },
  { "comp_info", T_OBJECT, offsetof(jpegObject,comp_info), 0,
    "Component info (List of Dictionaries)." },
  { "coef_arrays", T_OBJECT, offsetof(jpegObject,coef_arrays), 0,
    "Coefficient arrays." },
  { "quant_tables", T_OBJECT, offsetof(jpegObject,quant_tables), 0,
    "Quantisation Tables." },
  { "comments", T_OBJECT, offsetof(jpegObject,comments), 0,
    "JPEG Comments (List of Strings)." },
  { "filename", T_OBJECT, offsetof(jpegObject,filename), 0,
    "Filename." },
  { NULL }
} ;

static PyTypeObject
jpegObjectType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "jpegObject",               /* tp_name */
   sizeof(jpegObject),         /* tp_basicsize */
   0,                         /* tp_itemsize */
   (destructor)jpegDel,     /* tp_dealloc */
   0, /* tp_print */        0, /* tp_getattr */
   0, /* tp_setattr */      0, /* tp_compare */
   0, /* tp_repr */         0, /* tp_as_number */
   0, /* tp_as_sequence */  0, /* tp_as_mapping */
   0, /* tp_hash */         0, /* tp_call */
   0, /* tp_str */          0, /* tp_getattro */
   0, /* tp_setattro */     0, /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags*/
   "JPEG object to load and save JPEG files and give access to the JPEG "
   "representation of images, i.e. to the quantised DCT coefficients and "
   "quantisation tables. "
   "It is rather crude most non-essential metadata are discarded. "
   "Huffmann tables are also discarded, and new optimised tables are always "
   "recomputed when the image is saved.",        /* tp_doc */
   0, /* tp_traverse */    0,  /* tp_clear */
   0, /* tp_richcompare */ 0,  /* tp_weaklistoffset */
   0, /* tp_iter */        0,  /* tp_iternext */
   jpegObject_methods,         /* tp_methods */
   jpegObject_members,         /* tp_members */
   0, /* tp_getset */     0,  /* tp_base */
   0, /* tp_dict */       0,  /* tp_descr_get */
   0, /* tp_descr_set */  0,  /* tp_dictoffset */
   (initproc)jpegInit,  /* tp_init */
   0, /* tp_alloc */      0,  /* tp_new */
} ;
static struct PyModuleDef jpegmodule =
{
    PyModuleDef_HEAD_INIT,
    "jpegObject", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    jpegObject_methods
};

/* void initjpegObject(void)
{
  PyObject * mod ;
  import_array() ;
  mod = Py_InitModule3 ( "jpegObject", NULL, "JPEG Toolbox" ) ;
  if ( mod == NULL ) return ;
  jpegObjectType.tp_new = PyType_GenericNew ;
  if ( PyType_Ready( &jpegObjectType ) < 0 ) return ;
  Py_INCREF ( &jpegObjectType ) ;
  PyModule_AddObject( mod, "jpegObject", (PyObject*) &jpegObjectType ) ;
  return ; 
}*/

PyMODINIT_FUNC
PyInit_jpegObject(void)
{
  PyObject * mod ;
  import_array() ;
  mod = PyModule_Create(&jpegmodule);
  if ( mod == NULL ) return NULL ;
  jpegObjectType.tp_new = PyType_GenericNew ;
  if ( PyType_Ready( &jpegObjectType ) < 0 ) return NULL ;
  Py_INCREF ( &jpegObjectType ) ;
  PyModule_AddObject( mod, "jpegObject", (PyObject*) &jpegObjectType ) ;
  return mod; 
}


