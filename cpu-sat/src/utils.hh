#pragma once

#include <algorithm>

namespace utils
{
    void erase_if(auto& container, auto predicate)
    {
        container.erase(
            std::remove_if(container.begin(), container.end(), predicate),
            container.end());
    }

    bool contains(auto& container, auto elt)
    {
        return std::find(container.begin(), container.end(), elt)
            != container.end();
    }

} // namespace utils
