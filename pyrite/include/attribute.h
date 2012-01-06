/*
 * GEM attribute functions include
 *
 */

extern int
gem_getAttribute(/*@null@*/ gemAttrs *attr, int aindex, char **name, int *atype,
                 int *alen, int **integers, double **reals, char **string);

extern int
gem_retAttribute(/*@null@*/ gemAttrs *attr, /*@null@*/ char *name, int *aindex,
                 int *atype, int *alen, int **integers, double **reals, 
                 char **string);

extern int
gem_setAttribute(gemAttrs **attrx, /*@null@*/ char *name, int atype, int alen,
                 /*@null@*/ int *integers, /*@null@*/ double *reals, 
                 /*@null@*/ char *string);

extern void
gem_clrAttributes(gemAttrs **attrx);

