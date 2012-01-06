#include <stdio.h>
#include <stdlib.h>

#include "gem.h"

int
gem_getEdge(/*@unused@*/
            gemBRep *brep,              /* (in)  BRep pointer */
            int     edge,               /* (in)  edge index */
            double  tlimit[],           /* (out) t range (2) */
            int     nodes[],            /* (out) bounding node indices (2) */
            int     faces[],            /* (out) trimmed faces (2) */
            int     *nattr)             /* (out) number of attributes */
{
  tlimit[0] = edge;
  tlimit[1] = edge + 1;
  nodes[0]  = edge;
  nodes[1]  = edge + 1;
  faces[0]  = edge;
  faces[1]  = edge + 1;
  *nattr    = 3;

  return GEM_SUCCESS;
}


// pygem_getBrepAttr(gemptr brep,
//                   int etype,
//                   int eindex,
//                   int aindex)
// returns (name, value) where value is int,float,string, tuple of ints or floats
int
gem_getBRepAttr(/*@unused@*/
                gemBRep *brep,          /* (in)  BRep pointer */
                int     etype,          /* (in)  Topological entity type */
                int     eindex,         /* (in)  Topological entity index */
                int     aindex,         /* (in)  Attribute index */
                char    *name[],        /* (out) pointer to attribute name */
                int     *atype,         /* (out) Atrribute type */
                int     *alen,          /* (out) Attribute length */
                                        /*       one of: */
                int     *integers[],    /* (out) pointer to integers/bools */
                double  *reals[],       /* (out) pointer to doubles */
                char    *string[])      /* (out) pointer to string */
{
  static int    ivec[2];
  static double rvec[2];

  *integers = NULL;
  *reals    = NULL;
  *string   = NULL;

  if (aindex == GEM_INTEGER) {
    ivec[0]   = etype;
    ivec[1]   = eindex;
    *name     = "Attribute.1";
    *atype    = GEM_INTEGER;
    *alen     = 2;
    *integers = ivec;
  } else if (aindex == GEM_REAL) {
    rvec[0]   = etype;
    rvec[1]   = eindex;
    *name     = "Attribute.2";
    *atype    = GEM_REAL;
    *alen     = 2;
    *reals    = rvec;
  } else {
    *name     = "Attribute.3";
    *atype    = GEM_STRING;
    *alen     = 6;
    *string   = "String";
  }

  return GEM_SUCCESS;
}

// pygem_setBrepAttr(gemptr brep, int etype, int eindex, string name,
//                   value)
//  returns PyNone
//    value is same as above in pygem_getBrepAttr
int
gem_setBRepAttr(/*@unused@*/
                gemBRep *brep,          /* (in)  BRep pointer */
                /*@unused@*/
                int     etype,          /* (in)  Topological entity type */
                /*@unused@*/
                int     eindex,         /* (in)  Topological entity index */
                char    name[],         /* (in)  pointer to attribute name */
                int     atype,          /* (in)  Atrribute type */
                int     alen,           /* (in)  Attribute length */
                                        /*       provide the appropriate one: */
                /*@null@*/
                int     integers[],     /* (in)  integers/bools */
                /*@null@*/
                double  reals[],        /* (in)  doubles */
                /*@null@*/
                char    string[])       /* (in)  string */
{
  if (name == NULL) return GEM_NULLNAME;
  if ((atype != GEM_BOOL) && (atype != GEM_INTEGER) && (atype != GEM_REAL) &&
      (atype != GEM_STRING)) return GEM_BADTYPE;
  if (alen > 0) {
    if (atype == GEM_STRING) {
      if (string   == NULL) return GEM_NULLVALUE;
    } else if (atype == GEM_REAL) {
      if (reals    == NULL) return GEM_NULLVALUE;
    } else {
      if (integers == NULL) return GEM_NULLVALUE;
    }
  }
  return GEM_SUCCESS;
}


int
gem_getTessel(/*@unused@*/
              gemDRep *drep,            /* (in)  pointer to DRep */
              /*@unused@*/
              gemPair bface,            /* (in)  BRep/Face index in DRep */
              int     *nvrt,            /* (out) number of vertices */
              double  *xyz[],           /* (out) pointer to the vertices */
              double  *uv[],            /* (out) pointer to the vertex params */
              gemConn **conn)           /* (out) pointer to connectivity */
{
  static gemConn connect;
  static int    tris[6]  = {1,2,3, 2,3,4};
  static int    neigh[6] = {2,0,0, 0,0,1};
  
  // in py version should be ndarrays (3 x nvrt and 2 x nvrt)
  static double xyzs[12] = {0.0,0.0,0.0, 1.0,0.0,0.0, 1.0,1.0,0.0, 0.0,1.0,0.0};
  static double uvs[8]   = {0.0,0.0,     1.0,0.0,     1.0,1.0,     0.0,1.0};

  connect.nTris     = 2;
  connect.Tris      = tris;
  connect.tNei      = neigh;
  connect.nQuad     = 0;
  connect.Quad      = NULL;
  connect.qNei      = NULL;
  connect.meshSz[0] = 0;
  connect.meshSz[1] = 0;
  connect.nSides    = 0;
  connect.sides     = NULL;

  *nvrt = 4;
  *xyz  = xyzs;
  *uv   = uvs;
  *conn = &connect;

  return GEM_SUCCESS;
}



