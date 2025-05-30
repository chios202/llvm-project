// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03_UCHAR_H
#define _LIBCPP___CXX03_UCHAR_H

/*
    uchar.h synopsis // since C++11

Macros:

    __STDC_UTF_16__
    __STDC_UTF_32__

Types:

  mbstate_t
  size_t

size_t mbrtoc8(char8_t* pc8, const char* s, size_t n, mbstate_t* ps);     // since C++20
size_t c8rtomb(char* s, char8_t c8, mbstate_t* ps);                       // since C++20
size_t mbrtoc16(char16_t* pc16, const char* s, size_t n, mbstate_t* ps);
size_t c16rtomb(char* s, char16_t c16, mbstate_t* ps);
size_t mbrtoc32(char32_t* pc32, const char* s, size_t n, mbstate_t* ps);
size_t c32rtomb(char* s, char32_t c32, mbstate_t* ps);

*/

#include <__cxx03/__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#endif // _LIBCPP___CXX03_UCHAR_H
