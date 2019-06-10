"""
Condition 1:

If a given digit at index i makes a number between [1, 9] then number of ways to decode string [0: i] would include number of ways to decode [0: i - 1] string. If given digit is 0 then it doesn't correspond to any valid letter.

Condition 2:

If given digits at index i and previous digit at i - 1 make a number which is less than 27 and greater than 9, then number of ways to decode string [0: i] would also include number of ways to decode [0: i - 2]. If two digits are less than 10 i.e. 09, 05, we can not take it as valid encoding as no such encoding exists in encoding map for letters to numbers.

Scenario 0:

If first digit is 0 it means encoding is invalid as 0 doesn't correspond to any letter and there is no digit before 0. But we don't need to handle this case because when we'll move to next character, number of ways to decode [0: i - 1] would already be 0 and number of ways to to decode [0: i - 2] would not taken into consideration as previous digit is 0 and '0X' doesn't correspond to any letter. So number of ways for this next digit would also be zero and so on. So if a digit is zero and it has zero number of ways to decode then this zero propagates to all furthur calculations and we get zero answer eventually

Scenario 00:

If two consecutive digits are 0 at index i and i + 1 in the middle of encoded string then it's also an invalid encoding. The number of ways for string [0: i] could be non-zero, but for 0 at index i + 1, Condition 1 and Condition 2 (Because previous digit is zero as well) would not fulfil so that's why Number of ways for string [0: i + 1] would be zero. Now we have seen in `Scenario 0` that if a digit is zero and number of ways are also zero at this digit then this zero will propagates till the end and we'll get 0 number of ways eventually.

"""
class Solution:
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        a, b = 1, 1
        for index in range(len(s)):
            tmp = 0
            if 0 < int(s[index]) <= 9:
                tmp += b
            if index - 1 >= 0 and 9 < int(s[index - 1: index + 1]) <= 26:
                tmp += a
            a, b = b, tmp
        return b

    def numDecodingsWithExtraMemory(self, s):
        """
        :type s: str
        :rtype: int
        """
        cache = [0] * len(s)
        for index in range(len(s)):
            if 0 < int(s[index]) <= 9:
                cache[index] += cache[index - 1] if index - 1 >= 0 else 1
            
            if index - 1 >= 0 and 9 < int(s[index - 1: index + 1]) <= 26:
                cache[index] += cache[index - 2] if index - 2 >= 0 else 1
        
        return cache[-1]