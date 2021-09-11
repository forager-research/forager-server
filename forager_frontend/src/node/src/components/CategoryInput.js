import React, { forwardRef, useState, useMemo } from "react";
import {
  Popover,
  PopoverBody,
} from "reactstrap";

import { Typeahead, useToken, ClearButton } from "react-bootstrap-typeahead";
import cx from "classnames";

import isEqual from "lodash/isEqual";
import sortBy from "lodash/sortBy";
import uniqWith from "lodash/uniqWith";
import union from "lodash/union";

import NewModeInput from "./NewModeInput";

// TODO(mihirg): Combine with this same constant in other places
const BUILT_IN_MODES = [
  ["POSITIVE", "Positive"],
  ["NEGATIVE", "Negative"],
  ["HARD_NEGATIVE", "Hard Negative"],
  ["UNSURE", "Unsure"],
];

const InteractiveToken = forwardRef((
  { active, children, className, onRemove, tabIndex, ...props },
  ref
) => (
  <div
    {...props}
    className={cx('rbt-token', 'rbt-token-removeable', {
      'rbt-token-active': !!active,
    }, className)}
    ref={ref}
    tabIndex={tabIndex || 0}>
    {children}
    <ClearButton
      className="rbt-token-remove-button"
      label="Remove"
      onClick={onRemove}
      tabIndex={-1}
    />
  </div>
));

const StaticToken = ({ children, className, disabled, href }) => {
  const classnames = cx('rbt-token', {
    'rbt-token-disabled': disabled,
  }, className);

  if (href && !disabled) {
    return (
      <a className={classnames} href={href}>
        {children}
      </a>
    );
  }

  return (
    <div className={classnames}>
      {children}
    </div>
  );
};

const MyToken = forwardRef((props, ref) => {
  const [isOpen, setIsOpen] = useState(false);
  const allProps = useToken(props);
  const {
    disabled,
    active,
    innerId,
    onRemove,
    onValueClick,
    index,
    customValues,
    category,
    customModesByCategory,
    categoryDispatch,
    allowNewModes,
    deduplicateByCategory,
    populateAll,
  } = allProps;

  const target = document.getElementById(innerId);

  return (
    <>
      {!disabled ?
        <InteractiveToken {...allProps} ref={ref} id={innerId} /> :
        <StaticToken {...allProps} id={innerId} />}
      {target !== null && <Popover
        placement="top"
        isOpen={isOpen && !!(document.getElementById(innerId))}
        target={innerId}
        trigger="hover"
        toggle={() => setIsOpen(!isOpen)}
        fade={false}
      >
        <PopoverBody onClick={e => e.stopPropagation()}>
          {BUILT_IN_MODES.map(([value, name]) =>
            <div>
              <a
                href="#"
                onClick={(e) => onValueClick(value, index, e)}
                className={`rbt-token ${value}`}
              >
                {name.toLowerCase()}
              </a>
            </div>)}
          {customValues.map(name =>
            <div>
              <a
                href="#"
                onClick={(e) => onValueClick(name, index, e)}
                className="rbt-token CUSTOM"
              >
                {name.toLowerCase()}
              </a>
            </div>)}
          {allowNewModes && <div>
            <NewModeInput
              category={category}
              customModesByCategory={customModesByCategory}
              categoryDispatch={categoryDispatch}
            />
          </div>}
          {!deduplicateByCategory && <div>
            <a
              href="#"
              onClick={(e) => populateAll(index, e)}
              className="rbt-token ALL"
            >
              (All of the above)
            </a>
          </div>}
        </PopoverBody>
      </Popover>}
    </>
  );
});

const CategoryInput = ({
  id,
  customModesByCategory,
  categoryDispatch,
  className,
  selected,
  setSelected,
  innerRef,
  deduplicateByCategory,
  allowNewModes,
  ...props
}) => {
  let options = [];
  if (deduplicateByCategory) {
    for (const category of customModesByCategory.keys()) {
      if (!selected.some(s => s.category === category)) {
        options.push({category, value: BUILT_IN_MODES[0][0]});
      }
    }
  } else {
    for (const [category, custom_values] of customModesByCategory) {
      for (const value of [...BUILT_IN_MODES, ...custom_values]) {
        const proposal = {category, value: Array.isArray(value) ? value[0] : value};
        if (!selected.some(s => isEqual(s, proposal))) {
          options.push(proposal);
          break;
        }
      }
    }
  }

  const onChange = (selected) => {
    let newSelected = selected.map(s => {
      if (s.customOption) {  // added
        categoryDispatch({
          type: "ADD_CATEGORY",
          category: s.category,
        });
        return {category: s.category, value: BUILT_IN_MODES[0][0]};
      }
      return s;
    });
    newSelected = uniqWith(newSelected, deduplicateByCategory ?
                           ((a, b) => a.category === b.category) : isEqual);
    setSelected(newSelected);
  };

  const onValueClick = (value, index, e) => {
    let newSelected = [...selected];
    newSelected[index] = {...newSelected[index], value};
    onChange(newSelected);
    e.preventDefault();
  };

  const populateAll = (index, e) => {
    const category = selected[index].category;
    const before = selected.slice(0, index);
    const standard = BUILT_IN_MODES.map(([value]) => {return {category, value};});
    const custom = customModesByCategory.get(category).map(value => {return {category, value};});
    const after = selected.slice(index + 1);
    onChange([...before, ...standard, ...custom, ...after]);
    e.preventDefault();
  };

  const renderToken = (option, { onRemove, disabled, key }, index) => {
    const isCustom = !BUILT_IN_MODES.some(([value]) => value === option.value);
    return (
      <MyToken
        key={key}
        disabled={disabled}
        innerId={`${id}-rbt-token-${index}`}
        className={isCustom ? "CUSTOM" : option.value}
        onRemove={onRemove}
        option={option}
        index={index}
        onValueClick={onValueClick}
        populateAll={populateAll}
        customValues={customModesByCategory.get(option.category) || []}
        category={option.category}
        customModesByCategory={customModesByCategory}
        categoryDispatch={categoryDispatch}
        allowNewModes={allowNewModes}
        deduplicateByCategory={deduplicateByCategory}>
        {option.category}{isCustom ? ` (${option.value})` : ""}
      </MyToken>
    );
  };

  return (
    <Typeahead
      id={id}
      multiple
      className={`typeahead-bar ${className || ""}`}
      options={options}
      selected={selected}
      onChange={onChange}
      renderToken={renderToken}
      labelKey="category"
      newSelectionPrefix="New category: "
      ref={innerRef}
      {...props}
    />
  );
}

export default CategoryInput;
