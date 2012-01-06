/*
 * GEM memory functions include
 *
 */

extern /*@null@*/ /*@out@*/ /*@only@*/ void *gem_allocate(size_t nbytes);

extern /*@null@*/ /*@only@*/ void *gem_callocate(size_t nele, size_t size);

extern /*@null@*/ /*@only@*/ void *gem_reallocate(/*@null@*/ /*@only@*/
                                      /*@returned@*/ void *ptr, size_t nbytes);

extern /*@null@*/ /*@only@*/ char *gem_strdup(/*@null@*/ const char *str);


